"""
streaming_pipeline.py — PySpark Structured Streaming.
Consomme network-flows et ids-detections depuis Kafka.
Calcule des stats par fenêtre de 10s et écrit dans dashboard/data/spark_stats.json.

Lancé comme subprocess par core/spark_client.py.
"""
import json
import os
import threading
from datetime import datetime
from pathlib import Path

ROOT          = Path(__file__).resolve().parents[2]
DATA_DIR      = ROOT / "dashboard" / "data"
STATS_FILE    = DATA_DIR / "spark_stats.json"
CHECKPOINT    = DATA_DIR / "spark_checkpoint"
BOOTSTRAP     = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TRIGGER_SECS  = "10 seconds"

_stats: dict = {"flows": {}, "detections": {}}
_lock  = threading.Lock()


def _flush():
    with _lock:
        _stats["updated_at"] = datetime.now().isoformat()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        STATS_FILE.write_text(json.dumps(_stats, ensure_ascii=False), encoding="utf-8")


def _process_flows(batch_df, epoch_id):
    if batch_df.isEmpty():
        return
    rows = batch_df.collect()
    protocols: dict = {}
    total_dur = 0
    src_ips: set = set()
    for r in rows:
        p = r.protocol or "Unknown"
        protocols[p] = protocols.get(p, 0) + 1
        total_dur += r.duration_us or 0
        if r.src_ip:
            src_ips.add(r.src_ip)
    n = len(rows)
    with _lock:
        _stats["flows"] = {
            "flow_count":      n,
            "avg_duration_us": round(total_dur / n, 1) if n else 0,
            "unique_src_ips":  len(src_ips),
            "protocols":       protocols,
            "flows_per_sec":   round(n / 10.0, 2),
            "epoch":           epoch_id,
        }
    _flush()


def _process_detections(batch_df, epoch_id):
    if batch_df.isEmpty():
        return
    rows = batch_df.collect()
    labels: dict = {}
    sevs = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "INFO": 0}
    total_conf = 0.0
    for r in rows:
        lbl = r.label or "Unknown"
        labels[lbl] = labels.get(lbl, 0) + 1
        sev = r.severity or "INFO"
        if sev in sevs:
            sevs[sev] += 1
        total_conf += r.confidence or 0.0
    n = len(rows)
    benign  = labels.get("Benign", 0)
    attacks = n - benign
    top5 = sorted(
        [(k, v) for k, v in labels.items() if k != "Benign"],
        key=lambda x: -x[1],
    )[:5]
    with _lock:
        _stats["detections"] = {
            "total":          n,
            "attacks":        attacks,
            "critical_count": sevs["CRITICAL"],
            "high_count":     sevs["HIGH"],
            "medium_count":   sevs["MEDIUM"],
            "threat_rate":    round(attacks / n, 3) if n else 0.0,
            "avg_confidence": round(total_conf / n, 4) if n else 0.0,
            "top_labels":     top5,
            "severities":     sevs,
            "epoch":          epoch_id,
        }
    _flush()


def main():
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, from_json
    from pyspark.sql.types import (
        DoubleType, IntegerType, LongType,
        StringType, StructField, StructType,
    )

    spark = (
        SparkSession.builder
        .appName("IDS-Kafka-Streaming")
        .master("local[2]")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0",
        )
        .config("spark.hadoop.io.nativeio.enabled", "false")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    flow_schema = StructType([
        StructField("flow_id",     StringType()),
        StructField("timestamp",   StringType()),
        StructField("src_ip",      StringType()),
        StructField("dst_ip",      StringType()),
        StructField("dst_port",    IntegerType()),
        StructField("protocol",    StringType()),
        StructField("duration_us", LongType()),
    ])

    detection_schema = StructType([
        StructField("label",      StringType()),
        StructField("severity",   StringType()),
        StructField("confidence", DoubleType()),
        StructField("timestamp",  StringType()),
        StructField("src_ip",     StringType()),
        StructField("protocol",   StringType()),
    ])

    def kafka_source(topic: str):
        return (
            spark.readStream
            .format("kafka")
            .option("kafka.bootstrap.servers", BOOTSTRAP)
            .option("subscribe", topic)
            .option("startingOffsets", "latest")
            .option("failOnDataLoss", "false")
            .load()
        )

    flows = (
        kafka_source("network-flows")
        .select(from_json(col("value").cast("string"), flow_schema).alias("f"))
        .select("f.*")
    )

    detections = (
        kafka_source("ids-detections")
        .select(from_json(col("value").cast("string"), detection_schema).alias("d"))
        .select("d.*")
    )

    q_flows = (
        flows.writeStream
        .foreachBatch(_process_flows)
        .outputMode("append")
        .trigger(processingTime=TRIGGER_SECS)
        .option("checkpointLocation", str(CHECKPOINT / "flows"))
        .start()
    )

    q_detections = (
        detections.writeStream
        .foreachBatch(_process_detections)
        .outputMode("append")
        .trigger(processingTime=TRIGGER_SECS)
        .option("checkpointLocation", str(CHECKPOINT / "detections"))
        .start()
    )

    print("[IDS-Spark] Streaming démarré.", flush=True)
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
