package br.org.gsj.ml.scala.util

import javax.imageio.ImageIO


import java.awt.image.BufferedImage
import java.io.File

case class PixelInfo(pixel:Int,a: Int, r: Int, g: Int, b: Int, x: Int, y: Int)

object ImageUtils {
  
  def loadImageArray(path: String): Array[PixelInfo] = {
    ImageIO.setUseCache(false)
    val image = ImageIO.read(new File(path))
    
    // obtain width and height of image
    val w = image.getWidth
    val h = image.getHeight
    
    var array = Array[PixelInfo]()
    var cont = 1
    for (x <- 0 until w)
      for (y <- 0 until h){
        val argba = printPixelARGB(image.getRGB(x, y))
        //println(x+","+y)
        array = array :+ PixelInfo(image.getRGB(x, y),argba._1,argba._2,argba._3,argba._4,x,y)
        cont += 1
      }
      array  
    
  }
  
  def printPixelARGB(pixel: Int):(Int,Int,Int,Int) = {
    val alpha = (pixel >> 24) & 0xff;
    val red = (pixel >> 16) & 0xff;
    val green = (pixel >> 8) & 0xff;
    val blue = (pixel) & 0xff;
    (alpha,red,green,blue)
  }
  
  
  def generateImage(img: BufferedImage, image_array: Array[Byte],colors:Map[Int,Int]): BufferedImage = {
    // obtain width and height of image
    val w = img.getWidth
    val h = img.getHeight
    
    if ( w*h != image_array.size)
      throw new IllegalArgumentException("image array does not fit the provided image");
    

    // create new image of the same size
    val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    
   var s = 0 
    for (x <- 0 until w)
      for (y <- 0 until h){
        
           out.setRGB(x, y, colors(image_array(s).toInt)) 
           s+=1}


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

//  def test() {
//    // read original image, and obtain width and height
//    val photo1 = ImageIO.read(new File("src/main/resources/datasets/clustering/data/mri-images-data/mri-healthy-brain.png"))
//
//    val photo2 = phototest(photo1)
//    generateImage(photo1,null)
//
//    // save image to file "test.jpg"
//    ImageIO.write(photo2, "jpg", new File("src/main/resources/datasets/clustering/data/mri-images-data/mri-healthy-test.png"))
//  }

}