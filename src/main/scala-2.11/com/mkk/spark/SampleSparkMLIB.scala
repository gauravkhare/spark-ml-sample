
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

//Define schema based on RAW Data Format
object SampleSparkMLIB {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

  val sc_schema = StructType(Array(
    StructField("date_field", StringType, true),
    StructField("usrId", StringType, true),
    StructField("host", StringType, true),
    StructField("geo", StringType, true),
    StructField("singleClick", IntegerType, true),
    StructField("doubleClick", IntegerType, true),
    StructField("Revenue", DoubleType, true),
    StructField("gender", StringType, true)
  ))

  val spark = SparkSession.builder.master("local").appName("Click Stream Data Load").getOrCreate()

  // Scala Functions as UDF
  // Can have java jar file to invoke JAR implict custom library
  def getUserTokensUDF = udf[Integer, String] { s =>
    if (s.contains(".")) {
      s.split(".").length
    } else {
      0
    }
  }

  val upperUDF = udf { s: String => s.toUpperCase }


  // Read raw data
  val raw_data = spark.read
    .format("com.databricks.spark.csv")
    .option("header", "true")
    .schema(sc_schema)
    .load("src/main/resources/raw_data.csv")
    .dropDuplicates()


  raw_data.createOrReplaceTempView("raw_data_tbl")

  // Cleanse DF to strip duplicates based on Business Logic

  val main_data = spark.sql(
    s"""
            SELECT 
            first(date_field)  as event_date,
            usrId,
            host,
            gender,
            (Case WHEN  max(singleClick) = 0  THEN 1 ELSE (max(singleClick)+1) END) AS singleClick,
            (Case WHEN  max(doubleClick) = 0  THEN 1 ELSE (max(doubleClick)+1) END) AS doubleClick,
            max(Revenue) as revenue,
            (Case WHEN  geo = 'us'  THEN 1 ELSE 0 END) AS target_geo
            FROM 
            raw_data_tbl
            GROUP by
            usrId,
            date_field,
            gender,
            host,
            geo
            """).dropDuplicates()


  // TO setup DF in to MLIB compatible format
  val logdata = main_data.select(col("target_geo").as("label"), col("singleClick"), col("doubleClick"), col("revenue"), col("gender"))
  val logregdata = logdata.na.drop()

  // Deal with Categorical Columns
  val genderIndexer = new StringIndexer().setInputCol("gender").setOutputCol("genderIndex")
  val genderEncoder = new OneHotEncoder().setInputCol("genderIndex").setOutputCol("genderVec")

  // Assemble everything together to be ("label","features") format
  val assembler = (new VectorAssembler()
    .setInputCols(Array("singleClick", "doubleClick", "revenue", "genderVec"))
    .setOutputCol("features"))

  // Setup pipeline
  val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)

  val lr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)

  val pipeline = new Pipeline().setStages(Array(genderIndexer, genderEncoder, assembler, lr))

  // Fit the pipeline to training documents.
  val model = pipeline.fit(training)

  // Get Results on Test Set
  val results = model.transform(test)

  // For Metrics and Evaluation
  import org.apache.spark.mllib.evaluation.MulticlassMetrics

  import spark.implicits._
  // Need to convert to RDD to use this
  val predictionAndLabels = results.select(col("prediction"), col("label")).as[(Double, Double)].rdd

  // Instantiate metrics object
  val metrics = new MulticlassMetrics(predictionAndLabels)

  // Confusion matrix
  println("Confusion matrix:")
  println(metrics.confusionMatrix)

}
}
