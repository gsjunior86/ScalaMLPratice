package br.org.gsj.ml.spark.classification

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.VectorAssembler

object PoliticalAfifiliationRegressionTree {

  case class Politician(
    party:                                  String,
    handicapped_infants:                    String,
    water_project_cost_sharing:             String,
    adoption_of_the_budget_resolution:      String,
    physician_fee_freeze:                   String,
    el_salvador_aid:                        String,
    religious_groups_in_schools:            String,
    anti_satellite_test_ban:                String,
    aid_to_nicaraguan_contras:              String,
    mx_missile:                             String,
    immigration:                            String,
    synfuels_corporation_cutback:           String,
    education_spending:                     String,
    superfund_right_to_sue:                 String,
    duty_free_exports:                      String,
    export_administration_act_south_africa: String)

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().appName("regressionTree").master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val data_path = "src/main/resources/datasets/classification/house-votes/house-votes-84.data"

    val politicians_df = spark.sparkContext.textFile(data_path)
      .map({
        f =>
          val arr = f.split(",")
          Politician(arr(0), arr(1), arr(2), arr(3), arr(4), arr(5), arr(6), arr(7),
            arr(8), arr(9), arr(10), arr(11), arr(12), arr(13), arr(14), arr(15))
      }).toDF()

    val categorical_columns = Array(politicians_df.columns.filter(p => !p.equals("party")): _*)

    var pipeline_stages = Array[PipelineStage]()

    for (column <- categorical_columns) {
      val string_indexer = new StringIndexer().setInputCol(column).setOutputCol(column + "_index")
      val encoder = new OneHotEncoderEstimator().setInputCols(Array(string_indexer.getOutputCol))
        .setOutputCols(Array(column + "_classVec"))
      pipeline_stages = pipeline_stages :+ string_indexer
      pipeline_stages = pipeline_stages :+ encoder

 
    }
    
     val label_string_idx = new StringIndexer().setInputCol("party").setOutputCol("label")
      pipeline_stages = pipeline_stages :+ label_string_idx
      val vector_assembler_inputs = Array[String](categorical_columns.map(f => f + "_classVec"):_*)
      val vector_assembler = new VectorAssembler().setInputCols(vector_assembler_inputs).setOutputCol("features")
      pipeline_stages = pipeline_stages :+ vector_assembler

    val pipeline = new Pipeline
    pipeline.setStages(pipeline_stages)
    val pipeline_model = pipeline.fit(politicians_df)
    val label_column = "label"
    val congressional_voting_features_df = pipeline_model.transform(politicians_df)
    .select("features", label_column,"party")
   
    congressional_voting_features_df.show

  }

}