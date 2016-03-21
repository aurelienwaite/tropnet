import breeze.linalg.DenseVector

object Deleteme {
  println("Welcome to the Scala worksheet")       //> Welcome to the Scala worksheet
  
  val vec = DenseVector(838.887939,1.776103,-1.235571,1.633822,1.254425,-0.826263,-1.191154,-0.288734,-3.169744,-2.949199,0.888441,-0.120083,1.28909) :* 1.833434
                                                  //> Mar 20, 2016 1:28:26 PM com.github.fommil.jni.JniLoader liberalLoad
                                                  //| INFO: successfully loaded /var/folders/5k/fdmtk6nn531b66t36g3vrt3r0000gn/T/j
                                                  //| niloader8121583394876259463netlib-native_system-osx-x86_64.jnilib
                                                  //| vec  : breeze.linalg.DenseVector[Double] = DenseVector(1538.0456695525259, 3
                                                  //| .256367627702, -2.265337880814, 2.995504804748, 2.29990544545, -1.5148986771
                                                  //| 419999, -2.183902242836, -0.529374732556, -5.811516420896, -5.407161719366, 
                                                  //| 1.6288979363940002, -0.220164255022, 2.36346143506)
  
	println(vec)                              //> DenseVector(1538.0456695525259, 3.256367627702, -2.265337880814, 2.995504804
                                                  //| 748, 2.29990544545, -1.5148986771419999, -2.183902242836, -0.529374732556, -
                                                  //| 5.811516420896, -5.407161719366, 1.6288979363940002, -0.220164255022, 2.3634
                                                  //| 6143506)
  
}