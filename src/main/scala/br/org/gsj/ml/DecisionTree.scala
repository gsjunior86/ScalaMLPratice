package br.org.gsj.ml

import org.apache.spark.sql.SparkSession

object DecisionTree {
  
  def main(args: Array[String]): Unit = {
    
    val spark = SparkSession.builder()
    .appName("DecisionTreeSpark")
    .master("local[*]")
    .getOrCreate()
    
    val sc = spark.sparkContext
    
  }
  
}