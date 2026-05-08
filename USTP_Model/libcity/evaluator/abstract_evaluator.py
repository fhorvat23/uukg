class AbstractEvaluator(object):

    def __init__(self, config):
        raise NotImplementedError('evaluator not implemented')

    def collect(self, batch):
        """Collect a batch of evaluation inputs

        Args:
            batch(dict): input data"""
        raise NotImplementedError('evaluator collect not implemented')

    def evaluate(self):
        """Returns the evaluation results of all previously collected batches"""
        raise NotImplementedError('evaluator evaluate not implemented')

    def save_result(self, save_path, filename=None):
        """Save the evaluation results to the filename file under the save_path folder

        Args:
            save_path: save path
            filename: Save file name"""
        raise NotImplementedError('evaluator save_result not implemented')

    def clear(self):
        """Clear the evaluation information of the previously collected batch. It is suitable for clearing at the beginning of each evaluation to eliminate the influence of previous evaluation inputs."""
        raise NotImplementedError('evaluator clear not implemented')
