package br.org.gsj.spark.sql

import org.apache.spark.sql.SparkSession

object LoadData {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("Load Data").master("local[*]").getOrCreate()

    val sc = spark.sparkContext
    val sqlContext = spark.sqlContext

    val file_location = "src/main/resources/trivago/data.dat"

    val data_rdd = sc.textFile(file_location)

    //      val data_df = spark.read.orc(file_location)

    data_rdd.top(5).map { x =>
      var s = x.replaceAll("", "{")
       s = s.replaceAll("", "}}")
       s = s.replaceAll("", "{")
       s = s.replaceAll("", "}")
       s = s.replaceAll("", ",")
       s = s.replaceAll("", "}{")
       s = s.replaceAll("[^A-Za-z0-9-.{}]", ",")
      while (s.startsWith(",")
          || s.startsWith("{")
          || s.startsWith("}")) {
        s = s.substring(1, s.length())
      }
      s
    }
      .foreach(println)
      
      data_rdd.top(5).foreach(println)

  }

}