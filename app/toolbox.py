# %%
from typing import DefaultDict
import numpy as np

from tqdm.auto import tqdm
from onstart import logger

# %%
# with open('a.npy', 'rb') as f:
#     img_array = np.load(f)
#     img_contour = np.load(f)

# img_array.shape, img_contour.shape


def count_limit_img_contour(img_contour, sz=2, count_limit=5000):
    # -- %%
    img_shadow = img_contour * 0

    localizer = np.array([e+1 for e in range(np.count_nonzero(img_contour))])
    logger.debug('Found {} pixels in total'.format(len(localizer)))

    img_shadow[img_contour > 0] = localizer

    two_links = DefaultDict(set)

    shape = img_shadow.shape
    for i in tqdm(range(shape[0]), 'Count limit iter slices'):
        for j in range(shape[1]):
            for k in range(shape[2]):
                a = img_shadow[i, j, k]
                if a == 0:
                    continue

                neighbors = [e
                             for e in np.unique(img_shadow[max(i-sz, 0):i+sz+1, max(j-sz, 0):j+sz+1, max(k-sz, 0):k+sz+1])
                             if e > 0 and e != a]

                for b in neighbors:
                    two_links[a].add(b)
                    two_links[b].add(a)

    two_links
    logger.debug('Large dict of two_links has {}'.format(len(two_links)))

    # -- %%
    _two_links = dict(two_links)

    for nid in tqdm([e for e in sorted(_two_links)]):
        if nid not in _two_links:
            continue

        while len(_two_links[nid].intersection([e for e in _two_links])) > 1:
            _two_links[nid] = _two_links[nid].union(
                *[_two_links.pop(e, set()) for e in _two_links.get(nid, [])])

    logger.debug('Found raw cluster for {}'.format(len(_two_links)))
    logger.debug('The clusters have counts of {}'.format(
        sorted([len(e) for _, e in _two_links.items()], reverse=True)))

    # -- %%
    selects = [(a, b, len(b))
               for a, b in _two_links.items() if len(b) > count_limit]

    logger.debug(
        'Found count_limit selector for {}'.format(len(selects)))
    if len(selects) > 0:
        _localizer = localizer * 0
        for a, b, c in selects:
            _localizer[[e-1 for e in b]] = 1
            logger.debug('Apply count_limit of {} pixels'.format(c))

        _map = img_contour * 0
        _map[img_contour > 0] = _localizer

        new_img_contour = img_contour * _map

    else:
        new_img_contour = img_contour * 1

    logger.debug('The img_contour is mapped, the new shape is {}'.format(
        new_img_contour.shape))

    return new_img_contour

# %%
# _img = img_shadow.copy()
# shape = _img.shape
# points = []
# for i in tqdm(range(shape[0])):
#     for j in range(shape[1]):
#         for k in range(shape[2]):
#             v = _img[i, j, k]
#             if v == 0:
#                 continue
#             points.append((int(v), i, j, k))

# df = pd.DataFrame(points, columns=['v', 'x', 'y', 'z'])
# df

# %%
# for a in tqdm(_two_links):
#     for b in _two_links[a]:
#         df.loc[b-1, 'v'] = int(a)

# df['count'] = df['v'].map(lambda e: len(_two_links.get(e, [])))
# df

# # %%
# df['v'].unique().shape

# # %%
# fig = px.scatter_3d(df, x='x', y='y', z='z', color='count')

# fig.update_layout(dict(scene={'aspectmode': 'data'},
#                        title='Graph'))

# fig.data[0]['marker']['size'] = 1

# fig.show()

# # %%
