package br.org.gsj.graph.scala

import org.apache.spark.sql.SparkSession

object Betweeness {
  
  def main(args: Array[String]): Unit = {
    
    val spark = SparkSession.builder.appName("BetweenessTest").master("local[*]").getOrCreate()
    
  }
  
}