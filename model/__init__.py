from .encoders import MetaNetRNN, EncoderRNN, WrapperEncoderRNN
from .decoders import AttnDecoderRNN, DecoderRNN
from .full_model_wrappers import BaselineModel

def describe_model(net):
    if type(net.encoder) is MetaNetRNN:
        print('EncoderMetaNet specs:')
        print(' nlayers=' + str(net.encoder.nlayers))
        print(' embedding_dim=' + str(net.encoder.embedding_dim))
        print(' dropout=' + str(net.encoder.dropout_p))
        print(' bi_encoder=' + str(net.encoder.bi_encoder))
        print(' n_input_symbols=' + str(net.encoder.input_size))
        print(' n_output_symbols=' + str(net.encoder.output_size))
        print('')
    elif type(net.encoder) is EncoderRNN or type(net.encoder) is WrapperEncoderRNN:
        print('EncoderRNN specs:')
        print(' bidirectional=' + str(net.encoder.bi))
        print(' nlayers=' + str(net.encoder.nlayers))
        print(' hidden_size=' + str(net.encoder.embedding_dim))
        print(' dropout=' + str(net.encoder.dropout_p))
        print(' n_input_symbols=' + str(net.encoder.input_size))
        print('')
    else:
        print('Encoder type not found...')

    if type(net.decoder) is AttnDecoderRNN:
        print('AttnDecoderRNN specs:')
        print(' nlayers=' + str(net.decoder.nlayers))
        print(' hidden_size=' + str(net.decoder.hidden_size))
        print(' dropout=' + str(net.decoder.dropout_p))
        print(' n_output_symbols=' + str(net.decoder.output_size))
        print('')
    elif type(net.decoder) is DecoderRNN:
        print('DecoderRNN specs:')
        print(' nlayers=' + str(net.decoder.nlayers))
        print(' hidden_size=' + str(net.decoder.hidden_size))
        print(' dropout=' + str(net.decoder.dropout_p))
        print(' n_output_symbols=' + str(net.decoder.output_size))
        print("")
    else:
        print('Decoder type not found...')
