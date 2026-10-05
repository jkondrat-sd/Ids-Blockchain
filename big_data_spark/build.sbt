name         := "IDS-BigData-Pipeline"
version      := "1.0"
scalaVersion := "2.12.18"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"  % "3.5.1",
  "org.apache.spark" %% "spark-sql"   % "3.5.1",
  "org.apache.spark" %% "spark-mllib" % "3.5.1"
)

// sbt assembly : fat JAR exécutable via java -jar
assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case "reference.conf"              => MergeStrategy.concat
  case _                             => MergeStrategy.first
}

assembly / assemblyJarName := "ids-bigdata-pipeline_2.12-1.0.jar"
assembly / mainClass       := Some("Pipeline")
Compile / mainClass        := Some("Pipeline")
