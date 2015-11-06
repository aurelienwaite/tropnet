package com.sdl

import scala.language.implicitConversions


package object tropnet{

  import java.io.FileInputStream

  type NBest = IndexedSeq[Hypothesis]

  type NBestSet = Seq[NBest]

  val formatter = new java.text.DecimalFormat("0.######")

  class Dot[T](v1: Seq[T])(implicit n: Numeric[T]) {
    import n._ // import * operator
    def dot(v2: Seq[T]) = {
      require(v1.size == v2.size)
        (v1 zip v2).map{ Function.tupled(_ * _)}.sum
    }
  }

  implicit def toDot[T: Numeric](v1: Seq[T]) = new Dot(v1)




}

