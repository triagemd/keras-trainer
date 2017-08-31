import argparse

from .trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_spec', type=str, help='name of the base model spec')
    parser.add_argument('dictionary', type=str, help='comma separated list of class names')
    parser.add_argument('train_dataset_dir', type=str, help='location of the training dataset directory')
    parser.add_argument('val_dataset_dir', type=str, help='location of the validation dataset directory')
    parser.add_argument('output_model_dir', type=str, help='location of the training output model directory')
    parser.add_argument('output_logs_dir', type=str, help='location of the training output logs directory')

    for key, option in Trainer.OPTIONS.items():
        kwargs = {'type': option.get('type', str)}
        if 'default' in option:
            kwargs['default'] = option['default']
        parser.add_argument('--%s' % (key, ), **kwargs)
    args = parser.parse_args()

    dictionary = args.dictionary.split(',')
    trainer = Trainer(args.model_spec, dictionary, args.train_dataset_dir, args.val_dataset_dir, args.output_model_dir, args.output_logs_dir)
    trainer.run()
