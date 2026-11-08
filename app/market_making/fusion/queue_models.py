"""Execution-only queue selection, common to every quoting plugin."""
QUEUE_MODELS=('risk_adverse','cancel_50_50')


def configure_queue(asset,name):
    if name=='risk_adverse':return asset.risk_adverse_queue_model()
    if name=='cancel_50_50':
        # Native f(x)=x**0 gives back-probability 1/(1+1)=0.5.
        # Native code subtracts trade consumption first, then bounds allocation
        # by the actual available front/back quantities. No Bernoulli sampling.
        return asset.power_prob_queue_model(0.0)
    raise ValueError('Unsupported queue model: '+str(name))
