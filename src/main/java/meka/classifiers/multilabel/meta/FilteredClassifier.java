package meka.classifiers.multilabel.meta;

import java.util.*;

import weka.classifiers.*;
import weka.classifiers.lazy.*;
import weka.core.*;
import weka.filters.*;

import meka.classifiers.multilabel.*;
import meka.core.*;
import meka.filters.multilabel.*;

public class FilteredClassifier extends ProblemTransformationMethod {

	private Filter m_Filter;

	public Filter getFilter() {

		return m_Filter;
	}

	public void setFilter(Filter filter) {

		m_Filter = filter;
	}

	public String[] getOptions() {

		Vector<String> options = new Vector<String>();

		options.add("-F");
		
		String result = getFilter().getClass().getName();

		if(getFilter() instanceof OptionHandler) {

			result += " " + Utils.joinOptions(((OptionHandler)getFilter()).getOptions());
		}

		options.add(result);

		Collections.addAll(options, super.getOptions());

		return options.toArray(new String[0]);
	}

	public void setOptions(String[] options) throws Exception {

		String filterString = Utils.getOption('F', options);

		if(filterString.length() == 0) {

			filterString = m_Filter.getClass().getName();

			if(m_Filter instanceof OptionHandler) {

				filterString += " " + Utils.joinOptions(((OptionHandler)m_Filter).getOptions());
			}
		}

		String[] filterSpec = Utils.splitOptions(filterString);

		if(filterSpec.length == 0) {

			throw new IllegalArgumentException("Invalid filter specification string");
		}

		String filterName = filterSpec[0];
		filterSpec[0] = "";
		setFilter((Filter)Utils.forName(Filter.class, filterName, filterSpec));

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	public FilteredClassifier() throws Exception {

		m_Filter = new JaccardEmbedder();
		BR br = new BR();
		br.setClassifier(new IBk(10));
		setClassifier(br);
	}

	@Override
	public void buildClassifier(Instances instances) throws Exception {

		instances = new Instances(instances);

		m_Filter.setInputFormat(instances);
		instances = Filter.useFilter(instances, m_Filter);

		m_Classifier.buildClassifier(instances);
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		m_Filter.input(instance);
		m_Filter.batchFinished();

		Instance filtered = m_Filter.output();

		if(filtered == null) {

			return new double[instance.classIndex()];
		}
		else {

			return m_Classifier.distributionForInstance(filtered);
		}
	}

	public static void main(String args[]) throws Exception {
		ProblemTransformationMethod.evaluation(new FilteredClassifier(), args);
	}
}
