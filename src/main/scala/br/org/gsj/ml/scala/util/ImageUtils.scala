package br.org.gsj.ml.scala.util

import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.io.File

object ImageUtils {
  
  val colors:Map[Int,Int] = Map(
      2 -> 0x95FFDF, //cyan
      3 -> 0xFF3333, //red
      4 -> 0x0058B6, //blue
      5 -> 0xE2CE06, //yellow
      6 -> 0xDB06E2, //pink
      7 -> 0x67C82C, //green
      8 -> 0x8136DC, //purple
      9 -> 0x356F07, //darkgreen
      10 -> 0xE5A812 //orange
      )
  
  def main(args: Array[String]): Unit = {
      
    test
  }
  
  
  def generateImage(img: BufferedImage, image_array: Array[Byte]): BufferedImage = {
    // obtain width and height of image
    val w = img.getWidth
    val h = img.getHeight
    
    if ( w*h != image_array.size)
      throw new IllegalArgumentException("image array does not fit the provided image");
    

    // create new image of the same size
    val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    
  
    for (x <- 0 until w)
      for (y <- 0 until h){
        var s = y
        if(x != 0){
           s = ((y + 256) * x)}
       
        
           out.setRGB(x, y, colors(image_array(s).toInt)) }


    out
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
        out.setRGB(x, y, img.getRGB(x, y))

    // draw red diagonal line
//    for (x <- 0 until (h min w))
//      out.setRGB(x, x, 0xff0000)

    out
  }

  def test() {
    // read original image, and obtain width and height
    val photo1 = ImageIO.read(new File("src/main/resources/datasets/clustering/data/mri-images-data/mri-healthy-brain.png"))

    val photo2 = phototest(photo1)
    generateImage(photo1,null)

    // save image to file "test.jpg"
    ImageIO.write(photo2, "jpg", new File("src/main/resources/datasets/clustering/data/mri-images-data/mri-healthy-test.png"))
  }

}