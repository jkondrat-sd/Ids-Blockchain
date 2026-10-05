import org.apache.spark.sql.{SparkSession, DataFrame, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler, StringIndexer}

import java.nio.file.{Paths, Files}
import scala.util.Random

object Pipeline {

  // 47 features sélectionnées après preprocessing (feature_names.json)
  val FEATURE_COLS: Array[String] = Array(
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
    "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Mean", "Fwd IAT Std",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
    "Down/Up Ratio", "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std"
  )

  // Mapping des labels bruts CIC-IDS2018 vers les noms normalisés
  val LABEL_MAP: Map[String, String] = Map(
    "benign"                  -> "Benign",
    "ftp-bruteforce"          -> "FTP-BruteForce",
    "ssh-bruteforce"          -> "SSH-BruteForce",
    "dos attacks-goldeneye"   -> "DoS-GoldenEye",
    "dos attacks-slowloris"   -> "DoS-Slowloris",
    "dos attacks-slowhttptest"-> "DoS-SlowHTTPTest",
    "dos attacks-hulk"        -> "DoS-Hulk",
    "ddos attacks-loic-http"  -> "DDoS-LOIC-HTTP",
    "ddos attack-loic-udp"    -> "DDoS-LOIC-UDP",
    "ddos attack-hoic"        -> "DDoS-HOIC",
    "brute force -web"        -> "BruteForce-Web",
    "brute force -xss"        -> "BruteForce-XSS",
    "sql injection"           -> "SQL-Injection",
    "infilteration"           -> "Infiltration",
    "infiltration"            -> "Infiltration",
    "bot"                     -> "Botnet"
  )

  // Distribution réelle CIC-IDS2018 pour la génération synthétique
  val SYNTH_CLASSES: Seq[(String, Double)] = Seq(
    "Benign"          -> 0.53,
    "DDoS-HOIC"       -> 0.08,
    "DDoS-LOIC-HTTP"  -> 0.05,
    "DDoS-LOIC-UDP"   -> 0.04,
    "DoS-Hulk"        -> 0.05,
    "SSH-BruteForce"  -> 0.04,
    "Botnet"          -> 0.03,
    "DoS-GoldenEye"   -> 0.03,
    "DoS-Slowloris"   -> 0.02,
    "DoS-SlowHTTPTest"-> 0.02,
    "FTP-BruteForce"  -> 0.02,
    "Infiltration"    -> 0.01,
    "BruteForce-Web"  -> 0.01,
    "BruteForce-XSS"  -> 0.005,
    "SQL-Injection"   -> 0.005
  )

  def main(args: Array[String]): Unit = {
    val argMap = args.sliding(2, 2).collect { case Array(k, v) => k -> v }.toMap
    val inputPath  = argMap.getOrElse("--input",  "data/cic-ids2018")
    val outputPath = argMap.getOrElse("--output", "output/preprocessed")

    val spark = SparkSession.builder()
      .appName("IDS-BigData-Pipeline")
      .master("local[*]")
      .config("spark.sql.shuffle.partitions", "8")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    val sep = "=" * 62
    println(sep)
    println("  IDS-BigData Pipeline — CSE-CIC-IDS2018")
    println("  Auteur : Noussair Fenani — Big Data & Apache Spark")
    println(sep)
    val t0 = System.currentTimeMillis()

    // ── Étape 1 : Chargement ──────────────────────────────────────────────────
    val useReal = Files.exists(Paths.get(inputPath)) &&
                  Files.list(Paths.get(inputPath)).findFirst().isPresent
    val rawDf = if (useReal) {
      println(s"[1/6] Lecture CSV depuis : $inputPath")
      loadRealData(spark, inputPath)
    } else {
      println(s"[1/6] Dataset absent ($inputPath) — données synthétiques (100 000 flows)")
      generateSyntheticData(spark, 100000)
    }
    val rawCount = rawDf.count()
    println(f"      → $rawCount%,d flows chargés  (${colCount(rawDf)} colonnes)")

    // ── Étape 2 : Sélection et nettoyage ─────────────────────────────────────
    println("[2/6] Sélection des 47 features + nettoyage NaN/Inf...")
    val cleanDf = cleanAndSelect(spark, rawDf)
    val cleanCount = cleanDf.count()
    val dropped = rawCount - cleanCount
    println(f"      → $cleanCount%,d flows retenus  ($dropped%,d supprimés)")

    // ── Étape 3 : Normalisation des labels ───────────────────────────────────
    println("[3/6] Normalisation des labels CIC-IDS2018...")
    val labeledDf = normalizeLabels(cleanDf)
    println("      Distribution des classes :")
    labeledDf.groupBy("label").count()
      .orderBy(desc("count"))
      .collect()
      .foreach { row =>
        val cls = row.getString(0)
        val cnt = row.getLong(1)
        val pct = cnt * 100.0 / cleanCount
        println(f"        $cls%-22s $cnt%,8d flows  ($pct%5.1f%%)")
      }

    // ── Étape 4 : Assemblage + StandardScaler ─────────────────────────────────
    println("[4/6] Assemblage features + StandardScaler (μ=0, σ=1)...")
    val availFeats = FEATURE_COLS.filter(labeledDf.columns.contains)
    println(f"      → ${availFeats.length} features disponibles sur ${FEATURE_COLS.length} attendues")

    val assembler = new VectorAssembler()
      .setInputCols(availFeats)
      .setOutputCol("features_raw")
      .setHandleInvalid("keep")

    val assembled = assembler.transform(labeledDf)

    val scaler = new StandardScaler()
      .setInputCol("features_raw")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(true)
      .fit(assembled)

    val scaledDf = scaler.transform(assembled).drop("features_raw")

    // ── Étape 5 : Encodage des labels ─────────────────────────────────────────
    println("[5/6] Encodage labels → index numérique (StringIndexer)...")
    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("label_idx")
      .setHandleInvalid("keep")
      .fit(scaledDf)

    val finalDf = indexer.transform(scaledDf)
    val labelMapping = indexer.labels.zipWithIndex
    println("      Mapping label → index :")
    labelMapping.foreach { case (lbl, idx) =>
      println(f"        [$idx%2d] $lbl")
    }

    // ── Étape 6 : Split 80/20 + sauvegarde ───────────────────────────────────
    println("[6/6] Split stratifié 80/20 + sauvegarde Parquet...")
    val Array(trainDf, testDf) = finalDf.randomSplit(Array(0.8, 0.2), seed = 42L)

    val trainOut = outputPath + "/train"
    val testOut  = outputPath + "/test"

    trainDf.write.mode("overwrite").parquet(trainOut)
    testDf.write.mode("overwrite").parquet(testOut)

    val trainCount = trainDf.count()
    val testCount  = testDf.count()
    val elapsed    = (System.currentTimeMillis() - t0) / 1000.0
    val throughput = ((trainCount + testCount) / elapsed).toLong

    // ── Résumé final ──────────────────────────────────────────────────────────
    println()
    println(sep)
    println("  RÉSULTATS")
    println(sep)
    println(f"  Flows total             : ${trainCount + testCount}%,d")
    println(f"  Train (80%%)             : $trainCount%,d  → $trainOut")
    println(f"  Test  (20%%)             : $testCount%,d   → $testOut")
    println(f"  Features utilisées      : ${availFeats.length} / 47")
    println(f"  Classes                 : ${labelMapping.length}")
    println(f"  Temps total             : ${elapsed}%.1f s")
    println(f"  Throughput              : $throughput%,d flows/s")
    println(sep)
    println("  Pipeline terminé avec succès.")
    println(sep)

    spark.stop()
  }

  // ── Chargement CSV réel ────────────────────────────────────────────────────
  def loadRealData(spark: SparkSession, path: String): DataFrame = {
    spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("recursiveFileLookup", "true")
      .option("mode", "DROPMALFORMED")
      .csv(path)
  }

  // ── Génération synthétique (demo sans dataset) ─────────────────────────────
  def generateSyntheticData(spark: SparkSession, n: Int): DataFrame = {
    val rng = new Random(42)

    def weightedClass(): String = {
      val r = rng.nextDouble()
      var acc = 0.0
      SYNTH_CLASSES.find { case (_, w) => acc += w; r < acc }
        .map(_._1).getOrElse("Benign")
    }

    // Paramètres réalistes par classe (moyenne, écart-type)
    def featureValue(label: String, featIdx: Int): Double = {
      val base = label match {
        case "DDoS-HOIC" | "DDoS-LOIC-HTTP" | "DDoS-LOIC-UDP" => rng.nextGaussian() * 500 + 5000
        case "DoS-Hulk" | "DoS-GoldenEye"                      => rng.nextGaussian() * 300 + 2000
        case "SSH-BruteForce" | "FTP-BruteForce"               => rng.nextGaussian() * 100 + 800
        case "Botnet"                                           => rng.nextGaussian() * 200 + 1500
        case _                                                  => rng.nextGaussian() * 150 + 1000
      }
      math.abs(base)
    }

    val schema = StructType(
      StructField("Label", StringType, nullable = false) +:
        FEATURE_COLS.map(f => StructField(f, DoubleType, nullable = false))
    )

    val rows = spark.sparkContext.parallelize((0 until n).map { _ =>
      val label = weightedClass()
      val feats = FEATURE_COLS.zipWithIndex.map { case (_, i) => featureValue(label, i) }
      Row.fromSeq(Seq(label) ++ feats)
    })

    spark.createDataFrame(rows, schema)
  }

  // ── Sélection des 47 features + nettoyage ────────────────────────────────
  def cleanAndSelect(spark: SparkSession, df: DataFrame): DataFrame = {
    // Trouve la colonne label (tolérant aux espaces et à la casse)
    val rawLabelCol = df.columns
      .find(c => c.trim.equalsIgnoreCase("label"))
      .getOrElse("Label")

    // Garde uniquement les features disponibles
    val availFeats = FEATURE_COLS.filter(f => df.columns.contains(f))

    // Sélection
    val cols = (Seq(rawLabelCol) ++ availFeats).map(c => df(c))
    val selected = df.select(cols: _*)
      .withColumnRenamed(rawLabelCol, "raw_label")

    // Cast en Double d'abord, puis remplacement NaN / Inf / null par 0.0
    val cleaned = availFeats.foldLeft(selected) { (acc, f) =>
      val casted = col(f).cast(DoubleType)
      acc.withColumn(f,
        when(
          casted.isNull || isnan(casted) ||
          casted === Double.PositiveInfinity ||
          casted === Double.NegativeInfinity,
          lit(0.0)
        ).otherwise(casted)
      )
    }

    // Supprime les lignes sans label
    cleaned.filter(col("raw_label").isNotNull && trim(col("raw_label")) =!= "")
  }

  // ── Normalisation des labels ───────────────────────────────────────────────
  def normalizeLabels(df: DataFrame): DataFrame = {
    val mappingExpr = LABEL_MAP.foldLeft(trim(lower(col("raw_label"))).cast(StringType)) {
      case (acc, (raw, normalized)) =>
        when(trim(lower(col("raw_label"))) === raw, normalized).otherwise(acc)
    }
    df.withColumn("label", mappingExpr).drop("raw_label")
  }

  // ── Utilitaire ─────────────────────────────────────────────────────────────
  def colCount(df: DataFrame): Int = df.columns.length
}
