package br.org.gsj.ml.spark.clustering

import org.apache.spark.sql.SparkSession

object MriClassification {
  
  def main(args: Array[String]): Unit = {
    
    val spark = SparkSession.builder().appName("mriClass").master("local[*]").getOrCreate()
    
  }
  
}