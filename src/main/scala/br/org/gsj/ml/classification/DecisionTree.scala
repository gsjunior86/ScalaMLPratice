package br.org.gsj.ml.classification

import org.apache.spark.sql.SparkSession

import org.apache.spark.sql.types._
import org.apache.spark.sql.Row

object DecisionTree {
  
  def main(args: Array[String]): Unit = {
    
    val spark = SparkSession.builder()
    .appName("DecisionTreeSpark")
    .master("local[*]")
    .getOrCreate()
    
    val sc = spark.sparkContext
    
    val zoo_rdd = sc.textFile("src/main/resources/datasets/zoo.data")
    .map(x=>x.split(","))    
    .map(row => Row(row(0).toString(),row(1).toInt,row(2).toInt,row(3).toInt,row(4).toInt,row(5).toInt
        ,row(6).toInt,row(7).toInt,row(8).toInt,row(9).toInt,row(10).toInt,row(11).toInt,row(12).toInt,row(13).toInt
        ,row(14).toInt,row(15).toInt,row(16).toInt,row(17).toInt))
    
    
    val schema = StructType(
          Array(
                StructField("animal_name", StringType, false),
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
        
        
        
        val zoo_df = spark.createDataFrame(zoo_rdd, schema)
        
        zoo_df.show()
    
  }
  
}