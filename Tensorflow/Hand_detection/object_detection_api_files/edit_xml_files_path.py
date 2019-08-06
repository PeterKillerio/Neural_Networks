from xml.etree import ElementTree as et
import os

#print(os.listdir())

for file in os.listdir():

    if(file.endswith('xml')):
        photo_name = file[:-3]
        photo_name = photo_name + "jpg"
        print(photo_name)

        abs_path = os.path.abspath(photo_name)
        print(abs_path)


        chopped_abs = abs_path.split("\\")
        folder_name = chopped_abs[-2]

        tree = et.parse(file)
        tree.find('.//folder').text = folder_name
        tree.find('.//filename').text = photo_name
        tree.find('.//path').text = abs_path
        #tree.find('.//enddate').text = '1/1/2011'

        tree.write(file)

