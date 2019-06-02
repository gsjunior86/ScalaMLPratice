package br.org.gsj.ml.spark.linear_regression

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import vegas._

object BikeSharing {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("bikeSharing-spark")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext

    val bsDay_rdd = sc.textFile("src/main/resources/datasets/bike_sharing/day.csv").
      mapPartitionsWithIndex((idx, f) => if (idx == 0) f.drop(1) else f).map(f => f.split(","))

    val bsDay_df = bsDay_rdd.map(f =>
      BS_Scheme(
        f(0).toInt,
        f(1).toString(),
        f(2).toInt,
        f(3).toInt,
        f(4).toInt,
        f(5).toInt,
        f(6).toInt,
        f(7).toInt,
        f(8).toInt,
        f(9).toDouble,
        f(10).toDouble,
        f(11).toDouble,
        f(12).toDouble,
        f(13).toInt,
        f(14).toInt,
        f(15).toInt)).toDF()

    bsDay_df.show

    val plot = Vegas("Country Pop").
      withData(
        Seq(
          Map("country" -> "USA", "population" -> 314),
          Map("country" -> "UK", "population" -> 64),
          Map("country" -> "DK", "population" -> 80))).
        encodeX("country", Nom).
        encodeY("population", Quant).
        mark(Bar)
        
        println(plot.html.pageHTML("foo"))

  }

  case class BS_Scheme(
    instant:    Int,
    date:       String,
    season:     Int,
    yr:         Int,
    mnth:       Int,
    holiday:    Int,
    weekday:    Int,
    workingDay: Int,
    weathersit: Int,
    temp:       Double,
    atemp:      Double,
    hum:        Double,
    windspeed:  Double,
    casual:     Int,
    registered: Int,
    cnt:        Int)

}