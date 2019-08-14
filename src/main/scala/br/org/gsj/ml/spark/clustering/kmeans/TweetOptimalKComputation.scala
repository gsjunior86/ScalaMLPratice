package br.org.gsj.ml.spark.clustering.kmeans

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.Column

object TweetOptimalKComputation {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("TweetClassfication")
      .master("local[*]")
      .getOrCreate()

    val tweets_df = spark.read.option("delimiter", "|")
      .csv("src/main/resources/datasets/clustering/data/health-tweets/*.txt")
      .withColumnRenamed("_c0", "tweet_id")
      .withColumnRenamed("_c1", "date")
      .withColumnRenamed("_c2", "tweet_msg")

    val splitUDF = udf { s: String =>
      s.
        toLowerCase()
        .replaceAll("https?://\\S+\\s?", "")
        .replaceAll("^[\\s\\.\\d]+", "")
        .replaceAll("[0-9]", "")
        .replaceAll("\"", "")
        .replaceAll("\\,", "")
        .replaceAll("!", "")
        .replaceAll(":", "")
        .replaceAll("\\?", "")
        .replaceAll("\\.", "")
        .split("\\s+")
        .filter(p => p.length() > 2)
    }
    
    val tweetClean = udf { s: Seq[String] =>
      s.map(f =>  f.replaceAll("'",""))
      .filterNot(p => p.startsWith("-") || p.startsWith("@")|| p.contains("&")
          || p.contains(";"))
    }
    
    val countArray = udf { s: Seq[String] =>
      s.length
    }

    val stop_words_array = spark.sparkContext
      .textFile("src/main/resources/datasets/clustering/data/health-tweets/stopwords/stopwords-en.txt")
      .collect()

    val stopwords = new StopWordsRemover().setInputCol("tweet_msg").setOutputCol("tweet_clean").setStopWords(stop_words_array)

    val new_df = tweets_df.withColumn("tweet_msg", splitUDF(col("tweet_msg")))
    new_df.printSchema()

    stopwords.transform(new_df).withColumn("tweet_clean", tweetClean(col("tweet_clean")))
    .withColumn("array_count", countArray(col("tweet_clean")))
     .drop("tweet_msg").select(col("tweet_id") +: (0 until 17).map(i => col("tweet_clean")(i).alias(s"col$i")): _* )
     
     .write.json("src/main/resources/datasets/clustering/data/health-tweets/json")

  }

}