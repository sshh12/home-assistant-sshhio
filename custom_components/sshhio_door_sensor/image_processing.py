"""Component that will detect door states."""

import logging
import os
import io
import cv2
import numpy as np

from homeassistant.core import split_entity_id
from homeassistant.const import CONF_ENTITY_ID, CONF_NAME
from homeassistant.components.image_processing import (
    ImageProcessingEntity,
    CONF_ENTITY_ID,
    CONF_NAME,
    CONF_SOURCE,
    PLATFORM_SCHEMA
)


def setup_platform(hass, config, add_entities, discovery_info=None):
    entities = []
    for camera in config[CONF_SOURCE]:
        entities.append(
            DoorSensor(camera[CONF_ENTITY_ID], camera.get('name', 'default'))
        )
    add_entities(entities)


class DoorSensor(ImageProcessingEntity):

    def __init__(self, camera_entity, door_name=None):
        super().__init__()

        self._camera = camera_entity
        self._state = None

        self.data_path = os.path.join(os.path.dirname(__file__), 'data', door_name.lower())
        self.mask = cv2.imread(os.path.join(self.data_path, 'mask.jpg'), 0) / 255.
        self.ref_pics = []
        for fn in os.listdir(self.data_path):
            self.ref_pics.append(
                (('open' if 'open' in fn else 'closed'),
                cv2.imread(os.path.join(self.data_path, fn), 0) * self.mask)
            )

        self._name = "{0} {1} Door".format(split_entity_id(camera_entity)[1].title(), door_name.title())

    @property
    def camera_entity(self):
        return self._camera

    @property
    def name(self):
        return self._name

    @property
    def state(self):
        return self._state

    def process_image(self, image):

        img = cv2.imdecode(np.asarray(bytearray(image)), cv2.IMREAD_UNCHANGED)
        cv2.imwrite('temp2.jpg', img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) * self.mask
        cv2.imwrite('temp.jpg', img)

        min_dist = 9e9
        min_class = None

        f = open('temp.txt', 'a')
        f.write('-------\n')

        for classname, img2 in self.ref_pics:
            dist = np.sum(np.abs(img - img2))
            f.write('{} {} \n'.format(classname, dist))
            if dist < min_dist:
                min_dist = dist
                min_class = classname

        f.close()

        self._state = classname