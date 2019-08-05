package br.org.gsj.ml.spark.clustering

import java.io.ByteArrayInputStream
import java.io.File

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import java.awt.image.MemoryImageSource

import javax.imageio.ImageIO
import java.awt.Toolkit
import java.awt.image.BufferedImage

object MriClustering {
  
  case class Image(data: Byte)
  
  def main(args: Array[String]): Unit = {
    
    val spark = SparkSession.builder().appName("mriClass").master("local[*]").getOrCreate()
    val mri_healthy_brain_image = "src/main/resources/datasets/clustering/data/mri-images-data/mri-healthy-brain.png"
    
    val image_df = spark.read.format("image").load(mri_healthy_brain_image).select(col("image.*"))
    image_df.show
    image_df.printSchema
    import spark.implicits._
    
    val data = image_df.rdd.collect().map(f => f(5))
    
    val data_array: Array[Byte] = data(0).asInstanceOf[Array[Byte]]
    
    val transposed_df = spark.sparkContext.parallelize(data_array).map(f => Image(f)).toDF

    transposed_df.show
    
    val features_col = Array("data")
    val vector_assembler = new VectorAssembler()
    .setInputCols(features_col)
    .setOutputCol("features")
    
    val mri_healthy_brain_df = vector_assembler.transform(transposed_df).select("features")
    
    val k = 5
    val kmeans = new KMeans().setK(k).setSeed(12345).setFeaturesCol("features")
    val kmeans_model = kmeans.fit(mri_healthy_brain_df)   
    val kmeans_centers = kmeans_model.clusterCenters
    println("Cluster Centers --------")
    for(k <- kmeans_centers)
      println(k)
      
      
    val mri_healthy_brain_clusters_df = kmeans_model.transform(mri_healthy_brain_df)
    .select("features","prediction")
    
    val image_array = mri_healthy_brain_clusters_df.select("prediction").rdd.map(f => f.getAs[Int](0)).collect()
    println(image_array.size)
    
//    val bImage = ImageIO.read(new File("src/main/resources/datasets/clustering/data/mri-images-data/mri-healthy-brain_cluster.jpg"));
//    val bos = new ByteArrayOutputStream();
//    ImageIO.write(bImage, "jpg", bos );
//    val datai = bos.toByteArray();
    
//    val ims = new MemoryImageSource(256,256,data_array.map(f => f.toInt),0,256)
//    val image = Toolkit.getDefaultToolkit().createImage(ims)
//    
//    val bi = new BufferedImage(image.getWidth(null), image.getHeight(null), BufferedImage.TYPE_INT_RGB);
//    bi.getGraphics().drawImage(image,0,0, null);
//    ImageIO.write(bi, "JPG", new File("src/main/resources/datasets/clustering/data/mri-images-data/mri-healthy-brain_cluster.jpg"));

  //http://otfried.org/scala/image.html

  }
  
  def phototest(img: BufferedImage): BufferedImage = {
  // obtain width and height of image
  val w = img.getWidth
  val h = img.getHeight

  // create new image of the same size
  val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)

  // copy pixels (mirror horizontally)
  for (x <- 0 until w)
    for (y <- 0 until h)
      out.setRGB(x, y, img.getRGB(w - x - 1, y) & 0xffffff)
  
  // draw red diagonal line
  for (x <- 0 until (h min w))
    out.setRGB(x, x, 0xff0000)

  out
}
  
def test() {
  // read original image, and obtain width and height
  val photo1 = ImageIO.read(new File("photo.jpg"))
  
  val photo2 = phototest(photo1) 

  // save image to file "test.jpg"
  ImageIO.write(photo2, "jpg", new File("test.jpg"))
}
  
}