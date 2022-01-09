#!/usr/bin/python3

__author__ = 'Jan Kempeneers'

import paho.mqtt.client as mqtt
import time, json, yaml


class Mqtt:
    def __init__(self, config_file="/home/pi/programs/my_mqtt_module.yml"):
        self.config_file = config_file
        # interpreting the yaml config file
        with open(self.config_file, 'r') as self.f: # use full path to file
            try:
                self.cfg = yaml.safe_load(self.f)
                print(self.cfg["server"])
            except yaml.YAMLError as exc:
                print(exc)
        self.flag_connected = 0
        self.client = mqtt.Client()

    def get_time_interval(self):
        return self.cfg["time_interval"]

    def get_sensor_snr(self):
        return self.cfg["sensor_snr"]

    def get_yml_item(self, item):
        return self.cfg[item]

    def on_connect(self, client, userdata, flags, rc):
        ## global flag_connected
        self.flag_connected = 1
        print("connected to mqtt broker")

    def on_disconnect(self, client, userdata, rc):
        ## global flag_connected
        self.flag_connected = 0

    def client_initialize(self):
        self.server=self.cfg["server"]
        self.port=self.cfg["port"]
        self.username=self.cfg["username"]
        self.password=self.cfg["password"]
        if self.username != "":
            self.client.username_pw_set(self.username, self.password)
        # keep retrying until endpoint available and connection established
        while not self.flag_connected == 1:
            try:
                print("trying ", end="", flush=True)
                self.client.connect(self.server,self.port,60)
                time.sleep(0.5)
                self.client.loop()
            except:
                time.sleep(1)
                pass
        print("mqtt client connected")

    def publish(self, msg_out):
        self.topic=self.cfg["topic"]
        if self.flag_connected == 1:
        # Publish message
            self.client.publish(self.topic, msg_out)
            print(msg_out, end=" ", flush=True)
        else:
        # Wait to reconnect
            while not self.flag_connected == 1:
                try:
                    print("disconnected, trying to reconnect.")
                    self.client.connect(self.server,self.port,60)
                    time.sleep(0.5)
                    self.client.loop()
                except:
                    time.sleep(1)
                    pass

    def start(self):
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client_initialize()

# sending some values for when testing
def run():
    mqtt1 = Mqtt("my_mqtt_module.yml")
    mqtt1.start()
    value=0
    while True:
        msg = {"temperature": value}
        msg_out = json.dumps(msg)
        mqtt1.publish(msg_out)
        time.sleep(1)
        value += 1
    mqtt1.client.disconnect()

def main():
    run()

if __name__ == "__main__":
    main()