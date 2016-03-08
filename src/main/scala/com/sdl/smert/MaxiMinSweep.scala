package com.sdl.smert

import breeze.linalg.DenseMatrix
import com.sdl.BleuStats
import com.sdl.tropnet.Caetano.Neuron
import breeze.linalg.sum
import breeze.linalg.DenseVector
import scala.annotation.tailrec
import scala.collection.mutable.Buffer

object MaxiMinSweep {

  def maxiMinSweepLine(toDelete: Range)(activated: DenseMatrix[Float], projection: DenseMatrix[Float], bs: IndexedSeq[BleuStats]) = {

    val deactivated = activated.copy
    deactivated(toDelete, ::) := 0.0f
    val projectedActivated = projection * activated
    val projectedDeactivated = projection * deactivated

    val minksum = for (c <- 0 until activated.cols) yield {
      val deactivatedBS = if (sum(projectedDeactivated(::, c)) == 0.0) BleuStats.bad else bs(c).deleteActivations()
      val l = Seq((L(projectedActivated(::, c), c), bs(c)), (L(projectedDeactivated(::, c), c), deactivatedBS)).sortBy(_._1.m)
      val left = l(0)._1
      val right = l(1)._1
      //If both have the same gradient, then pick the one with the lowest y
      if (left.m == right.m) {
        val resorted = l.sortBy(_._1.y)
        //println(resorted)
        println(resorted.head)
        println(resorted.head._2.activated)
        println(resorted.last)
        //println(resorted(0))
        (resorted.head, resorted.head, c)
      } else {
        left.x = (right.y - left.y) / (left.m - right.m)
        /*val update = DenseVector(right.x + 1, 1).t * projection
        val activatedScore = update * activated(::,c)
        val deactivatedScore = update * deactivated(::,c)
        println(left)
        println(right)
        println(right.x)
        println(l(1)._2.activated == 1)
        println(s"activated: $activatedScore, deactivated: $deactivatedScore")*/
        //Swap the lines, because we want the minimum
        (l(1), l(0), c)
      }
    }
    for(m<-minksum) println(m)
    //println(minksum)

    val sorted = minksum.sortBy(_._2._1.x)
    // A bit hacky, but parallel lines are stored twice
    //TODO: Fix so if empty add bad
    val filtered = sorted.filter{case (left, right, _) => left!=right} 
    val initials = for (s <- sorted) yield (s._1)

    
    
    //println(filtered.size)

    def update(nbest: DenseMatrix[Float], index: Int, bs : BleuStats) = {
      val update =  if(bs.activated ==1) 
        projectedActivated(::, index)
      else
        projectedDeactivated(::, index)
      nbest(::, index) := update
      nbest
    }

    /*
     * Converts a sequence of L objects into a Matrix representation of the NBest
     */
    def nbest2Mat(nbest: IndexedSeq[(L, BleuStats)]) = {
      val nbestMat = DenseMatrix.zeros[Float](2, nbest.size)
      for (((l, bs), i) <- nbest.view.zipWithIndex) update(nbestMat, i, bs)
      (nbestMat, nbest.map(_._2))
    }

    /*
     * Creates a set of intervals. In each interval a hypothesis is minimal. The output is a set of N-best lists in matrix
     * form with the minimal representation
     */
    val minned = filtered.foldLeft(List((nbest2Mat(initials), Float.NegativeInfinity))) { (prev, s) =>
      val (_, (l, bs), index) = s
      val (nbest, _) = prev.last     
      val updated = nbest._1.copy 
      update(updated, index, bs)
      val bsUpdated = nbest._2.updated(index, bs)
      prev :+ ((updated, bsUpdated), l.x)
    }
    
    for( seq <- minned.sliding(2)) seq match { 
      case Seq(((m, b),start), ((_, _),end)) => {
        //println(s"${m.cols} ${m.rows} ${b.size} $start $end")
        val checked = if (start.isNegInfinity) end - 2 else start
        val point = DenseVector((checked + end) /2 , 1).t
        val activatedScores = (point * projectedActivated).t
        val deactivatedScores = (point * projectedDeactivated).t
        for( (((a, d), bs), i) <- (activatedScores.toScalaVector() zip deactivatedScores.toScalaVector() zip b).view.zipWithIndex) {
          if((a>d && bs.activated == 1) || (d>a && bs.activated == 0)) {
            println(s"$a $d ${bs.activated} ${m(::, i)} ${projectedActivated(::, i)} ${projectedDeactivated(::, i)}")
          }
        }
        
      }
      case Seq(((m, b),start)) => 
        println(s"${m.rows} ${b.size} $start")
    }

    @tailrec
    def sweepOverMinned(in: List[((DenseMatrix[Float], IndexedSeq[BleuStats]), Float)], accum: Buffer[(Float, BleuStats)]): Seq[(Float, Int)] = {
      in match {
        case Nil => Nil
        case head :: tail => {
          val ((mat, bs), interval) = head
          val swept = Sweep.sweepLine(mat)
          val nextInterval = tail match {
            case Nil         => Float.PositiveInfinity
            case (_, i) :: _ => i
          }
          @tailrec
          def filterSwept(toFilter: List[(Float, Int)], accum: Buffer[(Float, Int)]): Unit = toFilter match {
            case Nil => Unit
            case head :: tail => {
              val (sweepInterval, _) = head
              val sweepNextInterval = tail match {
                case Nil         => Float.PositiveInfinity
                case (e, _) :: _ => e
              }
              if ((sweepInterval >= interval && sweepInterval < nextInterval) ||
                (sweepInterval <= interval && sweepNextInterval > nextInterval)) accum += head
              filterSwept(tail, accum)
            }
          }
          val filtered = Buffer[(Float, Int)]()
          filterSwept(swept.toList, filtered)
          val filteredWithBs = filtered.map {
            case (interval, hypIndex) =>
              (interval, bs(hypIndex))
          }
          //println(s"$interval, $nextInterval")
          //println(swept)
          for (h <- filtered.headOption) {
            val first = (interval, bs(h._2))
            accum append first
            accum ++ filtered.tail
          }
          sweepOverMinned(tail, accum)
        }
      }
    }

    val swept = Buffer[(Float, BleuStats)]()
    sweepOverMinned(minned, swept)
    val merged = swept.tail.foldLeft(Buffer[(Float, BleuStats)](swept.head)) { (accum, next) =>
      if (accum.last._2 != next._2) accum += next
      accum
    }
    val res = merged.toIndexedSeq
    //println(res)
    res
  }

}