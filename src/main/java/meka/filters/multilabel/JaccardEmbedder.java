package meka.filters.multilabel;

import java.util.*;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.*;
import weka.classifiers.trees.*;
import weka.core.*;
import weka.filters.*;

import meka.classifiers.multilabel.*;
import meka.classifiers.multitarget.*;
import meka.core.*;

public class JaccardEmbedder extends SimpleBatchFilter {

	Classifier m_EmbedderTemplate;
	Classifier[] m_Embedder;
	int m_Dimensions;

	public Classifier getEmbedder() {

		return m_EmbedderTemplate;
	}

	public void setEmbedder(Classifier embedder) {

		m_EmbedderTemplate = embedder;
	}

	public int getDimensions() {

		return m_Dimensions;
	}

	public void setDimensions(int dimensions) {

		m_Dimensions = dimensions;
	}

	public String[] getOptions() {

		Vector<String> options = new Vector<String>();

		options.add("-D");
		options.add(Integer.toString(getDimensions()));

		options.add("-W");
		Classifier c = getEmbedder();
		String result = c.getClass().getName();

		if(c instanceof OptionHandler) {

			result += " " + Utils.joinOptions(((OptionHandler)c).getOptions());
		}

		options.add(result);

		Collections.addAll(options, super.getOptions());

		return options.toArray(new String[options.size()]);
	}

	public void setOptions(String[] options) throws Exception {

		String dimensionsString = Utils.getOption("D", options);

		if(dimensionsString.length() == 0) {

			dimensionsString = "16";
		}

		setDimensions(Integer.parseInt(dimensionsString));

		String embedderString = Utils.getOption("W", options);

		if(embedderString.length() == 0) {
			
			embedderString = getEmbedder().getClass().getName();

			if(getEmbedder() instanceof OptionHandler) {

				embedderString += " " + Utils.joinOptions(((OptionHandler)getEmbedder()).getOptions());
			}
		}

		String[] embedderSpec = Utils.splitOptions(embedderString);

		if(embedderSpec.length == 0) {

			throw new IllegalArgumentException("Invalid filter specification string");
		}

		String embedderName = embedderSpec[0];
		embedderSpec[0] = "";
		setEmbedder((Classifier)Utils.forName(Classifier.class, embedderName, embedderSpec));

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	public JaccardEmbedder() throws Exception {

		m_EmbedderTemplate = new RandomForest();

		m_Dimensions = 16;
	}

	@Override
	public String globalInfo() {

		return "Embeds the instances into R^n such that Euclidean distance is a good estimator of the Jaccard distance between the labels. An ensemble of regression models is used to perform the embedding.";
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat) throws Exception {

		ArrayList<Attribute> attributes = new ArrayList<Attribute>();

		for(int i = 0; i < inputFormat.classIndex(); i++) {

			attributes.add(inputFormat.attribute(i));
		}

		for(int i = 0; i < m_Dimensions; i++) {

			attributes.add(new Attribute("target" + Integer.toString(i)));
		}

		Instances newInstances = new Instances(inputFormat.relationName(), attributes, 0);
		newInstances.setClassIndex(inputFormat.classIndex());

		return newInstances;
	}

	@Override
	protected Instances process(Instances instances) throws Exception {

		if(!isFirstBatchDone()) {

			train(instances);
		}
		
		Instances predictions = predict(instances);

		return predictions;
	}

	private Instances reformatInput(Instances instances) throws Exception {

		ArrayList<Attribute> attributes = new ArrayList<Attribute>();

		for(int i = 0; i < m_Dimensions; i++) {

			attributes.add(new Attribute("target" + Integer.toString(i)));
		}

		for(int i = instances.classIndex(); i < instances.numAttributes(); i++) {

			attributes.add(instances.attribute(i));
		}

		Instances newInstances = new Instances(instances.relationName(), attributes, 0);
		newInstances.setClassIndex(m_Dimensions);

		for(int i = 0; i < instances.numInstances(); i++) {
			
			double[] values = new double[newInstances.numAttributes()];

			for(int j = instances.classIndex(); j < instances.numAttributes(); j++) {
				
				values[j - instances.classIndex() + m_Dimensions] = instances.instance(i).value(j);
			}

			newInstances.add(new DenseInstance(1, values));
		}

		return newInstances;
	}

	private double[][] computeTargets(Instances instances) throws Exception {

		double[][] targets = new double[instances.numInstances()][m_Dimensions];
		double[][] targetMeans = new double[instances.numInstances()][m_Dimensions];
		double[][] targetVars = new double[instances.numInstances()][m_Dimensions];
		double[] numUpdates = new double[instances.numInstances()];
		double[] iGrad = new double[m_Dimensions];
		double[] jGrad = new double[m_Dimensions];
		int[] iLabels = new int[instances.classIndex()];
		int[] jLabels = new int[instances.classIndex()];
		
		//Initialise targets
		Random rng = new Random();

		for(int i = 0; i < targets.length; i++) {

			numUpdates[i] = 0;

			for(int j = 0; j < targets[i].length; j++) {

				targets[i][j] = rng.nextGaussian() * 0.02;
				targetMeans[i][j] = 0;
				targetVars[i][j] = 0;
			}
		}

		double loss = Double.MAX_VALUE;
		int numbad = 0;

		//while(numbad < 5) {
		for(int q = 0; q < 10000; q++) {

			double newLoss = 0;

			for(int i = 0; i < instances.numInstances(); i++) {

				//Select another instance
				int j = rng.nextInt(instances.numInstances());

				//Get the labels
				for(int k = 0; k < iLabels.length; k++) {

					iLabels[k] = (int)instances.instance(i).value(k);
					jLabels[k] = (int)instances.instance(j).value(k);
				}

				//Compute the Jaccard Distance between the two instances
				double y = 1.0 - Metrics.P_Accuracy(iLabels, jLabels);

				//Compute the current estimate for the Jaccard distance
				double yHat = 0;

				for(int k = 0; k < targets[i].length; k++) {
					
					yHat += Math.pow(targets[i][k] - targets[j][k], 2);
				}

				//Update the loss function
				double diff;
				/*newLoss += Math.pow(y - Math.min(yHat, 1), 2);

				if(y < 1.0) {

					diff = yHat - y;
				}
				else {


					if(yHat < 1.0) {

						diff = yHat - y;
					}
					else {

						diff = 0;
					}
				}*/

				/*newLoss += (1.0 - y) * yHat + y * Math.max(1.0 - yHat, 0.0);

				diff = (1.0 - y);

				if(yHat < 1.0)
				{
					diff -= y;
				}*/

				numUpdates[i] += 1.0;
				numUpdates[j] += 1.0;

				if(y < 1.0)
				{
					diff = (yHat - y) * 2.0;
					newLoss += Math.pow(y - yHat, 2);
				}
				else
				{
					newLoss -= Math.min(1.0, yHat);

					if(yHat < 1.0)
					{
						diff = -1.0;
					}
					else
					{
						diff = 0;
					}
				}

				//Compute the gradient w.r.t. the targets
				for(int k = 0; k < m_Dimensions; k++) {
					
					iGrad[k] = diff * (targets[i][k] - targets[j][k]);
					jGrad[k] = diff * (targets[j][k] - targets[i][k]);

					targetMeans[i][k] = targetMeans[i][k] * 0.9 + iGrad[k] * 0.1;
					targetMeans[j][k] = targetMeans[j][k] * 0.9 + jGrad[k] * 0.1;

					targetVars[i][k] = targetVars[i][k] * 0.999 + iGrad[k] * iGrad[k] * 0.001;
					targetVars[j][k] = targetVars[j][k] * 0.999 + jGrad[k] * jGrad[k] * 0.001;

					double lri = 0.01 * (Math.sqrt(1.0 - Math.pow(0.999, numUpdates[i])) / (1.0 - Math.pow(0.9, numUpdates[i])));
					double lrj = 0.01 * (Math.sqrt(1.0 - Math.pow(0.999, numUpdates[j])) / (1.0 - Math.pow(0.9, numUpdates[j])));

					targets[i][k] -= lri * targetMeans[i][k] / Math.sqrt(targetVars[i][k] + 1.0e-8);
					targets[j][k] -= lrj * targetMeans[j][k] / Math.sqrt(targetVars[j][k] + 1.0e-8);
				}
			}

			newLoss /= instances.numInstances();

			/*if(loss - newLoss < 1e-4)
			{
				break;
			}
			else
			{
				loss = newLoss;
			}*/
			if(newLoss < loss)
			{
				loss = newLoss;
				numbad = 0;
			}
			else
			{
				numbad++;
			}
		}

		double[] mean = new double[m_Dimensions];

		for(int i = 0; i < targets.length; i++) {

			for(int j = 0; j < m_Dimensions; j++) {

				mean[j] += targets[i][j] / (double)targets.length;
			}
		}

		for(int i = 0; i < targets.length; i++) {

			for(int j = 0; j < m_Dimensions; j++) {

				targets[i][j] -= mean[j];
			}
		}

		return targets;
	}

	private void train(Instances instances) throws Exception {

		double[][] targets = computeTargets(instances);
		Instances input = reformatInput(instances);

		for(int i = 0; i < input.numInstances(); i++) {

			for(int j = 0; j < m_Dimensions; j++) {

				input.instance(i).setValue(j, targets[i][j]);
			}
		}

		m_Templates = new Instances[m_Dimensions];
		m_Embedder = AbstractClassifier.makeCopies(m_EmbedderTemplate, m_Dimensions);

		for(int i = 0; i < m_Dimensions; i++) {

			Instances data = F.keepLabels(new Instances(input), m_Dimensions, new int[]{i});
			data.setClassIndex(0);
			m_Embedder[i].buildClassifier(data);
			m_Templates[i] = new Instances(data, 0);
		}
	}

	Instances[] m_Templates;

	private Instances predict(Instances instances) throws Exception {

		Instances input = reformatInput(instances);
		Instances embedded = determineOutputFormat(instances);

		for(int i = 0; i < input.numInstances(); i++) {

			double[] values = new double[embedded.numAttributes()];
			
			for(int j = 0; j < instances.classIndex(); j++) {

				values[j] = instances.instance(i).value(j);
			}

			for(int j = 0; j < m_Dimensions; j++) {

				Instance x = (Instance)input.instance(i).copy();
				x.setDataset(null);
				x = MLUtils.keepAttributesAt(x, new int[]{j}, m_Dimensions);
				x.setDataset(m_Templates[j]);

				values[j + instances.classIndex()] = m_Embedder[j].distributionForInstance(x)[0];
			}

			embedded.add(new DenseInstance(1, values));
		}

		return embedded;
	}
}
