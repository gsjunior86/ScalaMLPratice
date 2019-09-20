package br.org.gsj.ml.scala.nn

import scala.math._

case class SimpleNN(val w1: Double, val w2: Double, val bias: Double){
  

  def apply(m1: Double, m2: Double): Double ={
    sigmoid(w1*m1+w2*m2+bias)
  }
  
  protected def sigmoid(value: Double): Double = {
    1/(1 + exp(-value))
  }
  
  
}