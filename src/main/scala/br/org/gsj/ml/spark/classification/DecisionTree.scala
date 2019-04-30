package br.org.gsj.ml.spark.classification

import org.apache.spark.sql.SparkSession

import org.apache.spark.sql.types._
import org.apache.spark.sql.Row
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassificationModel


object DecisionTree {
  
  def main(args: Array[String]): Unit = {
    
    val spark = SparkSession.builder()
    .appName("DecisionTreeSpark")
    .master("local[*]")
    .getOrCreate()
    
    val sc = spark.sparkContext
    
    import spark.implicits._
    
    val zoo_rdd = sc.textFile("src/main/resources/datasets/zoo.data")
    .map(x=>x.split(","))    
    .map(row => Row(row(1).toInt,row(2).toInt,row(3).toInt,row(4).toInt,row(5).toInt
        ,row(6).toInt,row(7).toInt,row(8).toInt,row(9).toInt,row(10).toInt,row(11).toInt,row(12).toInt,row(13).toInt
        ,row(14).toInt,row(15).toInt,row(16).toInt,row(17).toInt))
    
    
    val schema = StructType(
          Array(
                StructField("hair", IntegerType, false),
                StructField("feathers", IntegerType, false),
                StructField("eggs", IntegerType, false),
                StructField("milk", IntegerType, false),
                StructField("airborne", IntegerType, false),
                StructField("aquatic", IntegerType, false),
                StructField("predator", IntegerType, false),
                StructField("toothed", IntegerType, false),
                StructField("backbone", IntegerType, false),
                StructField("breathes", IntegerType, false),
                StructField("venomous", IntegerType, false),
                StructField("fins", IntegerType, false),
                StructField("legs", IntegerType, false),
                StructField("tail", IntegerType, false),
                StructField("domestic", IntegerType, false),
                StructField("catsize", IntegerType, false),
                StructField("type", IntegerType, false)
              )
        )
        
        val dt = new DecisionTreeClassifier()
          .setLabelCol("Label")
          .setFeaturesCol("Features")
        
        var zoo_df = spark.createDataFrame(zoo_rdd, schema)

            
        val va = new VectorAssembler().setOutputCol("Features").
            setInputCols(Array("hair","feathers","eggs","milk","airborne","aquatic",
                "predator","toothed","backbone","breathes","venomous","fins","legs","tail","domestic","catsize"))
            
        val zoo_labeled = va.transform(zoo_df).select("Features","type").withColumnRenamed("type", "Label")
         
        // Split the data into training and test sets (30% held out for testing).
        val Array(trainingData, testData) = zoo_labeled.randomSplit(Array(0.7, 0.3))
        
        val pipeline = new Pipeline().setStages(Array(dt))
        
        val model = pipeline.fit(zoo_labeled)
        
        
        val predictions = model.transform(testData)
        
        // rows to display.
        predictions.show
        
        // Select (prediction, true label) and compute test error.
        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("Label")
          .setPredictionCol("prediction")
          .setMetricName("accuracy")
        val accuracy = evaluator.evaluate(predictions)
        println(s"Test Error = ${(1.0 - accuracy)}")
        
        val treeModel = model.stages(0).asInstanceOf[DecisionTreeClassificationModel]
        println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
    
  }
  
}