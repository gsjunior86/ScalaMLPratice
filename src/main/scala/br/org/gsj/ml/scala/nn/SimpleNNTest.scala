package br.org.gsj.ml.scala.nn

object SimpleNNTest {
  
  /**
   * 
   * Flower Classification
   * 
   * length, width , color
   * 
   */

  def main(args: Array[String]): Unit = {

    val data = List((3D, 1.5, 1D), (2D, 1D, 0D), (4D, 1.5, 1D), (3D, 1D, 0D),
      (3.5, .5, 1D), (2D, .5, 0D), (5.5, 1D, 1D), (1D, 1D, 0D))

    val w1 = scala.util.Random.nextDouble() * 2 - 1
    val w2 = scala.util.Random.nextDouble() * 2 - 1
    val bias = scala.util.Random.nextDouble() * 2 - 1

    println("w1: " + w1)
    println("w2: " + w2)
    println("bias: " + bias)

    for (b <- data) {
      println("Classficiation: " + SimpleNN(w1, w2, bias)(b._1, b._2))
      println()
    }

  }

}