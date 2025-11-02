# tracker.py
import numpy as np
from scipy.spatial import distance

class Tracker:
    """
    Simple centroid-based tracker.
    Keeps track of object centroids and assigns persistent IDs.
    """
    def __init__(self, max_disappeared=30, max_distance=60):
        # next unique object ID
        self.nextObjectID = 0
        # dict: objectID -> centroid (x, y)
        self.objects = {}
        # dict: objectID -> number of consecutive frames it has disappeared
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        """
        rects: list of bounding boxes [x1, y1, x2, y2]
        returns list of [x1, y1, x2, y2, objectID]
        """
        # If no detections, mark existing objects as disappeared
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return []

        # compute input centroids
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        # if no objects currently, register all
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(tuple(input_centroids[i]))
            # return matched rects with IDs
            return [ [*rects[i], i] for i in range(len(rects)) ]

        # build an array of current object centroids
        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())
        D = distance.cdist(np.array(objectCentroids), input_centroids)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()
        assignments = []

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.max_distance:
                continue
            objectID = objectIDs[row]
            self.objects[objectID] = tuple(input_centroids[col])
            self.disappeared[objectID] = 0
            usedRows.add(row)
            usedCols.add(col)
            assignments.append((objectID, col))

        # find unused rows -> increase disappeared counter for those objects
        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.max_disappeared:
                self.deregister(objectID)

        # find unused cols -> register new objects
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)
        for col in unusedCols:
            self.register(tuple(input_centroids[col]))

        # build output list of bounding boxes with assigned IDs
        results = []
        # Build mapping from centroid index to objectID
        centroid_to_id = {}
        for oid, cent_idx in assignments:
            centroid_to_id[cent_idx] = oid
        # For centroids that were newly registered, they have last assigned IDs (nextObjectID-1 etc.)
        # Build reverse mapping from objectCentroids to ID
        for oid, centroid in self.objects.items():
            # find matching centroid index if exists
            for idx, ic in enumerate(input_centroids):
                if tuple(ic) == tuple(centroid):
                    centroid_to_id[idx] = oid

        for i, rect in enumerate(rects):
            obj_id = centroid_to_id.get(i, None)
            if obj_id is None:
                # fallback: assign nearest object if close, else skip
                obj_id = -1
            results.append([*rect, obj_id])
        return results
