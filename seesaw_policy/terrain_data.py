def pack_terrain_data(data, out=None):
    """Write schema v1 directly; optional output is reusable rollout scratch."""
    n = data['heightmap']['height'].shape[0]
    reference=data['heightmap']['height']
    h_end=reference[0].numel()*2
    f_end=h_end+data['footmap']['height'][0].numel()*2
    r_end=f_end+data['spatial']['distance'].shape[1]*5
    width=r_end+data['body']['valid'].shape[1]*4
    if out is None:
        out=reference.new_empty(n,width)
    if out.shape!=(n,width) or out.device!=reference.device or out.dtype!=reference.dtype:
        raise ValueError('Packed terrain output shape/device/dtype mismatch')
    for group,start,end in (('heightmap',0,h_end),('footmap',h_end,f_end)):
        d = data[group]
        target=out[:,start:end].view(n,-1,2)
        target[...,0].copy_(d['height'].reshape(n,-1))
        target[...,1].copy_(d['valid'].reshape(n,-1))
    d = data['spatial']
    target=out[:,f_end:r_end].view(n,1974,5)
    target[...,0].copy_(d['distance'])
    target[...,1].copy_(d['valid'])
    target[...,2:].copy_(d['normal'])
    d = data['body']
    target=out[:,r_end:].view(n,24,4)
    target[...,:3].copy_(d['vector'])
    target[...,3].copy_(d['valid'])
    return out
