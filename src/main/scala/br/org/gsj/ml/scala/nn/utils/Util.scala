package br.org.gsj.ml.scala.nn.utils

object Util {
  
  
  def cost(prediction: Double,target : Double): Double = {
    Math.pow((prediction - target), 2)
  }
  
//  def num_slope(num: Double):Double ={
//    val h = 0.0001
//    
//  }
  
}