//read in raw data to dataframe
val data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("C:\\Users\\xdere\\Downloads\\AirCustom.csv")

//create a view to use SQL queries
data.createOrReplaceTempView("data_view")

//filtered dataframe with NULL values removed (-200 values)
val results = spark.sql("SELECT * FROM data_view WHERE CO_GT != -200 AND PT08_S1_CO != -200 AND NMHC_GT != -200 AND C6H6_GT != -200 AND PT08_S2_NMHC != -200 AND NOx_GT != -200 AND PT08_S3_NOx != -200 AND NO2_GT != -200 AND PT08_S4_NO2 != -200 AND PT08_S5_O3 != -200 AND T != -200 AND RH != -200 AND AH != -200 AND T_Next != -200")

//convert date column into a single number representation of month
//ex. 3/10/2004 -> 3
val results2 = results.withColumn("Date", substring_index(results("Date"), "/", 1)).withColumn("Time", substring_index(results("Time"), ":", 1))

//convert string type to int type for use in regression
import org.apache.spark.sql.functions._
val toInt    = udf[Int, String]( _.toInt)
val results3 = results2.withColumn("Date", toInt(results2("Date"))).withColumn("Time", toInt(results2("Time")))

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

//prepare features vector
val assembler = new VectorAssembler().setInputCols(Array("Date", "Time", "CO_GT", "PT08_S1_CO", "NMHC_GT", "C6H6_GT", "PT08_S2_NMHC", "NOx_GT", "PT08_S3_NOx", "NO2_GT", "PT08_S4_NO2", "PT08_S5_O3", "T", "RH", "AH" )).setOutputCol("features")

val output = assembler.transform(results3)

//split data into training and testing sets
val Array(trainingData, testData) = output.randomSplit(Array(0.7, 0.3))

//definte new decision tree regressor
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.Pipeline
import spark.implicits._

val dt = new DecisionTreeRegressor().setLabelCol("T_Next").setFeaturesCol("features")

//prepare pipeline for model
val pipeline = new Pipeline().setStages(Array(dt))
val model = pipeline.fit(trainingData)

//run run model on test data
val predictions = model.transform(testData)
predictions.select("prediction", "T_Next", "features").show(5)

//evaluate data with RMSE
val evaluator = new RegressionEvaluator().setLabelCol("T_Next").setPredictionCol("prediction").setMetricName("rmse")

val rmse = evaluator.evaluate(predictions)

//attempt Cross validation
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

val paramGrid = new ParamGridBuilder().addGrid(dt.maxDepth, Array(1,2,3,4,5,6)).addGrid(dt.minInfoGain, Array(0.0, 0.1, 0.2)).addGrid(dt.minInstancesPerNode, Array(1,2)).build()
val numFolds = 10
val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(numFolds)
val model = cv.fit(trainingData)
val predictions = model.transform(testData)
predictions.select("prediction", "T_Next", "features").show(5)
val rmse = evaluator.evaluate(predictions)