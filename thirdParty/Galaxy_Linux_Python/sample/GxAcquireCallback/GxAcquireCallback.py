# version:1.0.2103.9101
import gxipy as gx
import time
from PIL import Image


def capture_callback_color(raw_image):
    # print height, width, and frame ID of the acquisition image
    print("Frame ID: %d   Height: %d   Width: %d"
            % (raw_image.get_frame_id(), raw_image.get_height(), raw_image.get_width()))

    # get RGB image from raw image
    rgb_image = raw_image.convert("RGB")
    if rgb_image is None:
        print('Failed to convert RawImage to RGBImage')
        return

    # create numpy array with data from rgb image
    numpy_image = rgb_image.get_numpy_array()
    if numpy_image is None:
        print('Failed to get numpy array from RGBImage')
        return

    # show acquired image
    img = Image.fromarray(numpy_image, 'RGB')
    img.show()


def capture_callback_mono(raw_image):
    # print height, width, and frame ID of the acquisition image
    print("Frame ID: %d   Height: %d   Width: %d"
            % (raw_image.get_frame_id(), raw_image.get_height(), raw_image.get_width()))

    # create numpy array with data from raw image
    numpy_image = raw_image.get_numpy_array()
    if numpy_image is None:
        print('Failed to get numpy array from RawImage')
        return

    # show acquired image
    img = Image.fromarray(numpy_image, 'L')
    img.show()


def main():
    # print the demo information
    print("")
    print("-------------------------------------------------------------")
    print("Sample to show how to acquire mono or color image by callback "
          "and show acquired image.")
    print("-------------------------------------------------------------")
    print("")
    print("Initializing......")
    print("")

    # create a device manager
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num is 0:
        print("Number of enumerated devices is 0")
        return

    # open the first device
    cam = device_manager.open_device_by_index(1)

    # set exposure
    cam.ExposureTime.set(10000)

    if dev_info_list[0].get("device_class") == gx.GxDeviceClassList.USB2:
        # set trigger mode
        cam.TriggerMode.set(gx.GxSwitchEntry.ON)
    else:
        # set trigger mode and trigger source
        cam.TriggerMode.set(gx.GxSwitchEntry.ON)
        cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)

    # get data stream
    data_stream = cam.data_stream[0]

    # Register capture callback (Notice: Linux USB2 SDK does not support register_capture_callback)
    if cam.PixelColorFilter.is_implemented() is True:
        data_stream.register_capture_callback(capture_callback_color)
    else:
        data_stream.register_capture_callback(capture_callback_mono)

    # start data acquisition
    cam.stream_on()

    print('<Start acquisition>')
    time.sleep(0.1)

    # Send trigger command
    cam.TriggerSoftware.send_command()

    # Waiting callback
    time.sleep(1)

    # stop acquisition
    cam.stream_off()
    
    print('<Stop acquisition>')

    # Unregister capture callback
    data_stream.unregister_capture_callback()

    # close device
    cam.close_device()


if __name__ == "__main__":
    main()
