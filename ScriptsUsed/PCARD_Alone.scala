package main.scala.djgarcia

import java.io.PrintWriter
import java.io.File
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature._
import org.apache.spark.mllib.tree.PCARD

import org.apache.spark.mllib.evaluation.MulticlassMetrics

object runEnsembles {


  def main(arg: Array[String]) {

    //Basic setup
    val jobName = "MLlib ROS 140"

    //Spark Configuration
    val conf = new SparkConf().setAppName(jobName)
    val sc = new SparkContext(conf)
		// val pathHeader = "file:///home/spark/datasets/susy.header"
		val pathHeader = "hdfs://hadoop-master/user/datasets/master/higgs/higgs.header"
    //Load train and test
    // val pathTrain = "file:///home/spark/datasets/susy-10k-tra.data"
		val pathTrain = "hdfs://hadoop-master/user/datasets/master/higgs/higgsMaster-Train.data"
    val rawDataTrain = sc.textFile(pathTrain)

    // val pathTest = "file:///home/spark/datasets/susy-10k-tst.data"
		val pathTest = "hdfs://hadoop-master/user/datasets/master/higgs/higgsMaster-Test.data"
    val rawDataTest = sc.textFile(pathTest) // Se puede añadir ", 10" como parámetro para facilitar a los procesos el uso de ficheros grandes.

    //Load train and test with KeelParser

    val converter = new KeelParser(sc, pathHeader)
    val train = sc.textFile(pathTrain, 10).map(line => converter.parserToLabeledPoint(line)).persist
    val test = sc.textFile(pathTest, 10).map(line => converter.parserToLabeledPoint(line)).persist


		val trainROS = ROS(train, 1.0).persist
		val classInfo = trainROS.map(lp => (lp.label, 1L)).reduceByKey(_ + _).collectAsMap()

    //Class balance
		// val classInfo = train.map(lp => (lp.label, 1L)).reduceByKey(_ + _).collectAsMap()
    // se puede hacer también:
    // val classInfo = train.map(lp => (lp.label, 1L))
    // Aquí SPARK no ha hecho nada aún
    //  .reduceByKey(_ + _) // acción --> Spark trabaja. Se reduce por clave de forma separada, para así tener por un lado los ejemplos de la etiqueta 1, y por otro los de la etiqueta 0, y que sume todos los que hay los de una etiqueta por un lado y los de la otra por otro lado.
    // .collectAsMap() // Se va a todos los reduce y los junta para imprimirlos por pantalla.

		//HME-BD Noise Filter
/*
		val nTrees_hme = 100
		val maxDepthRF_hme = 10
		val partitions_hme = 4

		val hme_bd_model = new HME_BD(train, nTrees_hme, partitions_hme, maxDepthRF_hme, 48151623)

		val hme_bd = hme_bd_model.runFilter()

		hme_bd.persist()
*/
/*
		// FCNN Filter - Instance Selection
		val k = 3 //number of neighbors

		val fcnn_mr_model = new FCNN_MR(train, k)

		val fcnn_mr = fcnn_mr_model.runPR()

		fcnn_mr.persist()

		fcnn_mr.count()
*/
    //PCARD

    val cuts = 10
    val trees = 50

    val pcardTrain = PCARD.train(trainROS, trees, cuts)

    val pcard = pcardTrain.predict(test)

    val labels = test.map(_.label).collect()

    var cont = 0

    for (i <- labels.indices) {
      if (labels(i) == pcard(i)) {
        cont += 1
      }
    }

    val testAcc = cont / labels.length.toFloat

    println(s"Test Accuracy = $testAcc")
    val predsAndLabels = sc.parallelize(pcard).zipWithIndex.map { case (v, k) => (k, v) }.join(test.zipWithIndex.map { case (v, k) => (k, v.label) }).map(_._2)
		/*var predsAndLabels = test.map { point =>
			val prediction = modelRF.predict(point.features)
			(prediction, point.label)
		}*/
		//Metrics PCARD
    val metrics_pc = new MulticlassMetrics(predsAndLabels)
    val precision_pc = metrics_pc.precision
    val cm_pc = metrics_pc.confusionMatrix
		val negative_pc = cm_pc.toArray(0) + cm_pc.toArray(2)
		val positive_pc = cm_pc.toArray(1) + cm_pc.toArray(3)
		val TNR_pc = cm_pc.toArray(0) / negative_pc
		val TPR_pc = cm_pc.toArray(3) / positive_pc
		val res_pc = TNR_pc * TPR_pc
		println(s"TNR: $TNR_pc" )
		println(s"TPR: $TPR_pc" )
		println(s"Resultado: $res_pc" )


    //Write Results
    // Para guardar en el cluster
    // val writer_pc = new PrintWriter(new File("/home/spark/results_hmd_filter_pcard.txt"))
    val writer_pc = new PrintWriter(new File("/home/x76654048/results_fcnn_mr_pcard.txt"))
    writer_pc.write(
      "Precision: " + precision_pc + "\n" +
        "Confusion Matrix " + cm_pc + "\n"
    )
		writer_pc.write("TNR: " + TNR_pc + "\n")
		writer_pc.write("TPR: " + TPR_pc + "\n")
		writer_pc.write("Resultado: " + res_pc + "\n")
    writer_pc.close()

  }
	// ROS

	def ROS(train: RDD[LabeledPoint], overRate: Double): RDD[LabeledPoint] = {
		var oversample: RDD[LabeledPoint] = train.sparkContext.emptyRDD

		val train_positive = train.filter(_.label == 1)
		val train_negative = train.filter(_.label == 0)
		val num_neg = train_negative.count().toDouble
		val num_pos = train_positive.count().toDouble

		if (num_pos > num_neg) {
		  val fraction = (num_pos * overRate) / num_neg
		  oversample = train_positive.union(train_negative.sample(withReplacement = true, fraction))
		} else {
		  val fraction = (num_neg * overRate) / num_pos
		  oversample = train_negative.union(train_positive.sample(withReplacement = true, fraction))
		}
		oversample.repartition(train.getNumPartitions)
	}


	// RUS

	def RUS(train: RDD[LabeledPoint]): RDD[LabeledPoint] = {
		var undersample: RDD[LabeledPoint] = train.sparkContext.emptyRDD

		val train_positive = train.filter(_.label == 1)
		val train_negative = train.filter(_.label == 0)
		val num_neg = train_negative.count().toDouble
		val num_pos = train_positive.count().toDouble

		if (num_pos > num_neg) {
		  val fraction = num_neg / num_pos
		  undersample = train_negative.union(train_positive.sample(withReplacement = false, fraction))
		} else {
		  val fraction = num_pos / num_neg
		  undersample = train_positive.union(train_negative.sample(withReplacement = false, fraction))
		}
		undersample
	}
}
