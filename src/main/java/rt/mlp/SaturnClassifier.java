package rt.mlp;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import rt.utils.DownloaderUtility;
import rt.utils.PlotUtil;

import java.io.File;
import java.util.concurrent.TimeUnit;

// build a multilayer perceptron using dl4j
/**
 * "Saturn" Data Classification Example
 *
 * Based on the data from Jason Baldridge:
 * 	https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots)
 *
 */


public class SaturnClassifier {


    public DataSetIterator[] initData(int batchSize) throws Exception {
        String dataLocalPath = DownloaderUtility.CLASSIFICATIONDATA.Download();

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(dataLocalPath, "saturn_data_train.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(dataLocalPath, "saturn_data_eval.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2);

        return new DataSetIterator[]{trainIter,testIter};
    }

    public MultiLayerConfiguration buildModel(double learningRate,int numInputs,int numOutputs,int numHiddenNodes){

        int seed = 123;

        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();
    }

    public MultiLayerNetwork trainModel(MultiLayerConfiguration conf,DataSetIterator trainIter,int nEpochs){
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));    //Print score every 10 parameter updates

        model.fit(trainIter, nEpochs);
        return model;
    }

    public void displayEvaluation(MultiLayerNetwork model,DataSetIterator testIter){
        System.out.println("Evaluate model....");
        Evaluation eval = model.evaluate(testIter);
        System.out.println(eval.stats());
        System.out.println("\n****************Evaluation finished********************");
    }

    public void generateVisuals(MultiLayerNetwork model, DataSetIterator trainIter, DataSetIterator testIter) throws Exception {

        double xMin = -15;
        double xMax = 15;
        double yMin = -15;
        double yMax = 15;

        //Let's evaluate the predictions at every point in the x/y input space, and plot this in the background
        int nPointsPerAxis = 100;

        //Generate x,y points that span the whole range of features
        INDArray allXYPoints = PlotUtil.generatePointsOnGraph(xMin, xMax, yMin, yMax, nPointsPerAxis);
        //Get train data and plot with predictions
        PlotUtil.plotTrainingData(model, trainIter, allXYPoints, nPointsPerAxis);
        TimeUnit.SECONDS.sleep(3);
        //Get test data, run the test data through the network to generate predictions, and plot those predictions:
        PlotUtil.plotTestData(model, testIter, allXYPoints, nPointsPerAxis);
    }

}
