import json
import base64
import argparse
import binascii

from .trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    for key, option in Trainer.OPTIONS.items():
        if 'type' in option and option['type'] is None:
            continue
        kwargs = {'type': option.get('type', str)}
        if 'default' in option:
            kwargs['default'] = option['default']
        parser.add_argument('--%s' % (key, ), **kwargs)
    args = parser.parse_args()

    options = vars(args)
    try:
        options['model_spec'] = json.loads(base64.b64decode(options['model_spec']))
    except binascii.Error:
        pass
    except TypeError:
        pass

    trainer = Trainer(**options)
    trainer.run()
