from models import deeplab, dinknet, lednet, hrnetv2, ocrnet


def getModel(config):
    if config['model_name'] == 'deeplab':
        print('-----    loading deeplabv3+ with %s as backbone and %s as enhance    -----' % (config['backbone'], config['enhance']))
        model = deeplab.DeepLab(in_channels=config['in_channels'], backbone=config['backbone'], enhance=config['enhance'],
                                output_stride=config['output_stride'], num_classes=config['num_classes'])
        print('******   load deeplabv3+ complete    ******')
    elif config['model_name'] == 'dinknet':
        print('-----    loading dinknet with %s as backbone    -----' % config['backbone'])
        model = dinknet.get_dink_model(in_channels=config['in_channels'], num_classes=config['num_classes'],
                                       backbone=config['backbone'])
        print('******   load dinknet complete   *******')
    elif config['model_name'] == 'lednet':
        print('-----   loading lednet   -------')
        model = lednet.LEDNet(in_channels=config['in_channels'], num_classes=config['num_classes'])
        print('******   load lednet complete   *******')
    elif config['model_name'] == 'ocrnet':
        print('-----   loading ocrnet   -------')
        model = ocrnet.get_ocr_model(in_channels=config['in_channels'], num_classes=config['num_classes'])
        print('******   load ocrnet complete   *******')
    elif config['model_name'] == 'hrnet':
        print('------   loading hrnetv2 which outputs 1/4 results   -----')
        model = hrnetv2.HRNetv2_ORG(in_channels=config['in_channels'], num_classes=config['num_classes'],
                                    cfg_name=config['config_name'])
        print('******   load hrnetv2 complete   *******')
    elif config['model_name'] == 'hrnet_up':
        print('-----   loading hrnetv2 with Up-Sample   -----')
        model = hrnetv2.HRNetv2_UP(in_channels=config['in_channels'], num_classes=config['num_classes'],
                                   cfg_name=config['config_name'])
        print('******   load hrnetv2 with Up-Sample complete   ******')
    elif config['model_name'] == 'hrnet_duc':
        print('-----   loading hrnetv2 with DUC as decoder   -----')
        model = hrnetv2.HRNetv2_DUC(in_channels=config['in_channels'], num_classes=config['num_classes'],
                                    cfg_name=config['config_name'])
        print('******   load hrnetv2 with DUC complete    ******')
    else:
        raise NotImplementedError
    return model
