package br.org.gsj.ml.scala.nn

import scala.math._

object SimpleNN {
  
  private val w1 : Double = .5
  private val w2 : Double = .2
  private val bias : Double = .3
  
  def compute(m1: Double, m2: Double): Double ={
    sigmoid(w1*m1+w2*m2+bias)
  }
  
  protected def sigmoid(value: Double): Double = {
    1/(1 + exp(-value))
  }
  
  
}