from omero.gateway import BlitzGateway, MapAnnotationWrapper
import numpy as np
import lavlab
#
import lavlab
lavlab.ctx.resources.io_max_threads = 2

import omero.gateway
import numpy as np
import threading
from typing import Generator, Optional
import lavlab
from skimage.feature import blob_dog

def get_rgb_tiles(  # pylint: disable=R0914
    img: omero.gateway.Image,
    tiles: list[tuple[int, int, int, tuple[int, int, int, int]]],
    res_lvl: Optional[int] = None,
    rps_bypass: bool = True,
    conn: omero.gateway.BlitzGateway = None,
) -> Generator[
    tuple[np.ndarray, tuple[int, int, int, tuple[int, int, int, int]]], None, None
]:
    """Pull tiles from omero faster using a ThreadPoolExecutor and executor.map!
 
    Parameters
    ----------
    img : omero.gateway.Image
        Omero image.
    tiles : list of tuple[int, int, int, tuple[int, int, int, int]]
        List of tiles to pull.
    res_lvl : int, optional
        Resolution level to pull, defaults to None.
    rps_bypass : bool, optional
        Passthrough to rps bypass option, defaults to True.
    conn : omero.gateway.BlitzGateway, optional
        Omero blitz gateway if not using omero image object, defaults to None.
 
    Yields
    ------
    tuple[np.ndarray, tuple[int, int, int, tuple[int, int, int, int]]]
        Tile and coords.
    """
    with lavlab.ctx.resources.io_pool as tpe:
        if conn is None:
            conn = img._conn  # pylint: disable=W0212
        local = threading.local()
 
        def work(args):
            """Runs inside a thread pool to get multiple tiles at a time."""
            pix_id, zct, coord, res_lvl, rps_bypass = args
            if getattr(local, "rps", None) is None:
                # Need to prepare a thread-specific rps
                local.rps = conn.c.sf.createRawPixelsStore()
                local.rps.setPixelsId(pix_id, rps_bypass)
                if res_lvl is None:
                    res_lvl = local.rps.getResolutionLevels()
                    res_lvl -= 1
                local.rps.setResolutionLevel(res_lvl)
            z, c, t = zct
            raw_data = []
            for c in range(3):
                raw_data.append(local.rps.getTile(z,c,t, *coord))
            return raw_data, (*zct, coord)
 
        def cleanup():
            """Cleans out the raw pixels stores after work is done."""
            if hasattr(local, "rps"):
                local.rps.close()
                delattr(local, "rps")
 
        try:
            # Use executor.map for streamlined processing
            pix_id = img.getPrimaryPixels().getId()
            tiles = tiles[0]
            args_iter = ((pix_id, (z, c, t), coord, res_lvl, rps_bypass) for z, c, t, coord in tiles)
            for raw_data, (z, c, t, coord) in tpe.map(work, args_iter):
                processed_data = np.zeros((coord[3], coord[2],3), dtype=np.uint8)
                for i, data in enumerate(raw_data):
                    np_data = np.frombuffer(data, dtype=np.uint8).reshape(coord[3], coord[2])
                    processed_data[:,:,i] = np_data
                yield processed_data, (z, c, t, coord)
        finally:
            # Cleanup resources
            for _ in range(tpe._max_workers):  # pylint: disable=W0212
                cleanup()
