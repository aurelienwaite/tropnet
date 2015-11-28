package com.sdl.smert

import com.sdl.smert.Sweep.sweepLine
import com.sdl.smert.SMERT.nbestToMatrix
import org.scalatest._
import org.scalatest.Matchers
import scala.io.Source
import com.sdl.NBest
import breeze.linalg.DenseMatrix


/**
 * @author aaw35
 */
class SweepLineSpec extends FlatSpec with Matchers{

  val DELTA = 0.001f
  "For the test NBest, SweepLine" should "find the same interval boundaries as lmert" in {
    val source = Source.fromURL(getClass.getResource("/nbest.txt"))
    val nBest = NBest.loadUCamNBest(source)
    val projection = DenseMatrix(
        (1f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f ),
        (1.000000f,0.820073f,1.048347f,0.798443f,0.349793f,0.286489f,15.352371f,-5.753633f,-3.766533f,0.052922f,0.624889f,-0.015877f)
    )
    val sweptIntervals = sweepLine(nbestToMatrix(nBest), projection).map(_._1)
    val lmertIntervals = Array[Float](Float.NegativeInfinity, -1.3284541f, -0.903594f, -0.85727775f, -0.77947056f, -0.26694396f, -0.1196983f, -0.056937158f, 1.1183509f)
    sweptIntervals.length should be(lmertIntervals.length)
    for(interval <- sweptIntervals zip lmertIntervals) {
      val (swept, lmert) = interval
      swept should === (lmert +- DELTA)
    }
  }
  
  "For the fire vector of the NBest, SweepLine" should "find 52.2 BLEU in the centre" in {
    val source = Source.fromURL(getClass.getResource("/fire.txt"))
    val nBest = NBest.loadUCamNBest(source)
    val a = DenseMatrix(
        (0f,1f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f ),
        (0f,1.000000f,0.820073f,1.048347f,0.798443f,0.349793f,0.286489f,15.352371f,-5.753633f,-3.766533f,0.052922f,0.624889f,-0.015877f)
    )
    val b = DenseMatrix(
        (0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f ),
        (0f,1.000000f,0.820073f,1.048347f,0.798443f,0.349793f,0.286489f,15.352371f,-5.753633f,-3.766533f,0.052922f,0.624889f,-0.015877f)
    )
    val projection = DenseMatrix.horzcat(a,b)
    val sweptIntervals = sweepLine(nbestToMatrix(nBest), projection).map(_._1) 
    println(sweptIntervals)
  }
  
  
}
