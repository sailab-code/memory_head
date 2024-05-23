from competitors.er import ERReservoir, ERRandom
from competitors.gdumb import GDumb
from competitors.ensemble import Ensemble
from competitors.mir import MIR
from competitors.agem import AGEM
import torch


class CompetitorFactory:
    @staticmethod
    def createCompetitor(options, net, input_size, num_classes):
        model_name = options.competitor
        buffer_size = options.buffer_size
        if model_name == "ER_reservoir":
            return ERReservoir(net, buffer_size, input_size)
        elif model_name == "ER_random":
            return ERRandom(net, buffer_size, input_size)
        elif model_name == "GDumb":
            return GDumb(net, buffer_size, input_size)
        elif model_name == "Ensemble":
            return Ensemble(net, options, num_classes)
        elif model_name == "MIR":
            return MIR(net, buffer_size, input_size, options)
        elif model_name == "AGEM":
            return AGEM(net, buffer_size, input_size, options)
        else:
            raise NotImplementedError

