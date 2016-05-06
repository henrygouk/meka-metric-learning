package meka.filters.multilabel;

import java.util.*;

import weka.core.*;
import weka.filters.*;

import meka.core.*;

public class LinearJaccardEmbedder extends SimpleBatchFilter {

	private int m_Dimensions;
	private double[][] m_Transform;

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

		Collections.addAll(options, super.getOptions());

		return options.toArray(new String[options.size()]);
	}

	public void setOptions(String[] options) throws Exception {

		String dimensionString = Utils.getOption("D", options);

		if(dimensionString.length() == 0) {

			dimensionString = "16";
		}

		setDimensions(Integer.parseInt(dimensionString));

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	public LinearJaccardEmbedder() throws Exception {

		m_Dimensions = 16;
	}

	@Override
	public String globalInfo() {

		return "Embeds the instances into R^n such that Euclidean distance is a good estimator of the Jaccard distance between the labels. A linear projection is used to perform the embedding.";
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

		return predict(instances);
	}

	private void train(Instances instances) {

		int numFeatures = instances.numAttributes() - instances.classIndex();

		Random rng = new Random();
		m_Transform = new double[m_Dimensions][numFeatures];

		for(int i = 0; i < m_Dimensions; i++) {

			for(int j = 0; j < numFeatures; j++) {

				m_Transform[i][j] = rng.nextGaussian() * 0.02;
			}
		}

		double[] iFeatures = new double[numFeatures];
		double[] jFeatures = new double[numFeatures];
		double[] iVec = new double[m_Dimensions];
		double[] jVec = new double[m_Dimensions];
		int[] iLabels = new int[instances.classIndex()];
		int[] jLabels = new int[instances.classIndex()];

		double loss = Double.MAX_VALUE;

		for(int q = 0; q < 1000; q++) {

			double newLoss = 0;

			for(int i = 0; i < instances.numInstances(); i++) {

				int j = rng.nextInt(instances.numInstances());

				for(int k = 0; k < iLabels.length; k++) {

					iLabels[k] = (int)instances.instance(i).value(k);
					jLabels[k] = (int)instances.instance(j).value(k);
				}

				for(int k = 0; k < iFeatures.length; k++) {

					iFeatures[k] = instances.instance(i).value(k + instances.classIndex());
					jFeatures[k] = instances.instance(j).value(k + instances.classIndex());
				}

				double y = 1.0 - Metrics.P_Accuracy(iLabels, jLabels);
				double yHat = 0;
				
				for(int k = 0; k < m_Dimensions; k++) {

					iVec[k] = MatrixUtils.dot(m_Transform[k], iFeatures);
					jVec[k] = MatrixUtils.dot(m_Transform[k], jFeatures);
					yHat += Math.pow(iVec[k] - jVec[k], 2);
				}

				double diff;

				newLoss += Math.pow(y - Math.min(yHat, 1.0), 2);

				if(y < 1.0) {

					diff = Math.min(yHat, 1.0) - y;
				}
				else {

					if(yHat < 1.0) {

						diff = yHat - y;
					}
					else {
						
						diff = 0;
					}
				}

				for(int k = 0; k < m_Dimensions; k++) {

					for(int l = 0; l < numFeatures; l++) {

						m_Transform[k][l] -= 0.00001 * instances.instance(i).value(l + instances.classIndex()) * (iVec[k] - jVec[k]) * diff;
						m_Transform[k][l] -= 0.00001 * instances.instance(j).value(l + instances.classIndex()) * (jVec[k] - iVec[k]) * diff;
					}
				}
			}

			System.out.println(newLoss / instances.numInstances());
		}
	}

	private Instances predict(Instances input) throws Exception {

		Instances embedded = determineOutputFormat(input);

		for(int i = 0; i < input.numInstances(); i++) {

			double[] features = new double[input.numAttributes() - input.classIndex()];

			for(int j = 0; j < features.length; j++) {

				features[j] = input.instance(i).value(j + input.classIndex());
			}

			double[] values = new double[m_Dimensions + input.classIndex()];

			for(int j = 0; j < input.classIndex(); j++) {

				values[j] = input.instance(i).value(j);
			}

			for(int j = 0; j < m_Dimensions; j++) {

				values[j + input.classIndex()] = MatrixUtils.dot(m_Transform[j], features);
			}

			embedded.add(new DenseInstance(1, values));
		}

		return embedded;
	}
}
