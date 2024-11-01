import struct
import os
import sys

PI = 3.14159
vector_num = 4
float_size = 4



import struct

def read_floats(file, count):
    return struct.unpack('f' * count, file.read(float_size * count))

def write_pov_file_and_generate_frame(data, num_objects, number):
    file_name = f"temp_{number}.pov"

    height = 100
    width = 100
    with open(file_name, 'w') as pov_file:
        pov_file.write(f"// {file_name} used in POV-Ray render\n\n")
        pov_file.write('#include "colors.inc"\n')
        pov_file.write('camera {\n\torthographic\n\tlocation <100,100,80>\n\trotate<0,-12,0>\n\tlook_at <0,10,10>\n\tsky <0,1,0>\n\tright <16/9,0,0>*50\n\tup <0,1,0>*50\n}\n')
        pov_file.write('light_source{<100,100,25> color White}\n')
        pov_file.write('light_source{<100,10,-40> color White}\n')
        pov_file.write('light_source{<20,-5,35> color White}\n\n')
        pov_file.write('#default {\n\ttexture{\n\t\tpigment {rgb 1}\n\t\tfinish {\n\t\t\tambient 0.01\n\t\t\tdiffuse 0.55\n\t\t\tspecular 0.9 roughness 0.001\n\t\t\treflection {0.0 1.0 fresnel on }\n\t\t\tconserve_energy\n\t\t}\n\t}\n}\n')
        pov_file.write('#macro Raster(RScale, HLine)\n\tpigment{ gradient x scale RScale\n\t\tcolor_map{[0.000   color rgbt<1,1,1,1>*0.4]\n\t\t\t[0+HLine color rgbt<1,1,1,1>*0.4]\n\t\t\t[0+HLine color rgbt<1,1,1,1>]\n\t\t\t[1-HLine color rgbt<1,1,1,1>]\n\t\t\t[1-HLine color rgbt<1,1,1,1>*0.4]\n\t\t\t[1.000   color rgbt<1,1,1,1>*0.4]} }\n\tfinish { ambient 0.15 diffuse 0.85}\n #end// of Raster(RScale, HLine)-macro\n//-------------------------------------------------------------------------\n')
        pov_file.write('#declare RasterScale = 1.0/2;\n#declare RasterHalfLine  = 0.035;\n#declare RasterHalfLineZ = 0.035;\n\n')
        pov_file.write('plane { <1,0,0>, -100    // plane with layered textures\n\ttexture { pigment{color rgbt<1,1,1,0.2>}\n\t\tfinish {ambient 1.0 diffuse 0.55}}\n//\ttexture { Raster(RasterScale,RasterHalfLine ) rotate<0,0,0> }\n//\ttexture { Raster(RasterScale,RasterHalfLineZ) rotate<0,90,0>}\n\trotate<0,0,0>\n\tno_shadow\n\t}\n\n')
        pov_file.write('#macro Axis_( AxisLen, Dark_Texture,Light_Texture)\nunion{\n\tcylinder { <0,-AxisLen,0>,<0,AxisLen,0>,0.05\n\t\ttexture{checker texture{Dark_Texture }\n\t\t\ttexture{Light_Texture}\n\t\t translate<3.7,10,10>}\n\t}\n\tcone{<0,AxisLen,0>,0.2,<0,AxisLen+0.7,0>,0\n\t\ttexture{Dark_Texture}\n\t\t}\n\t} // end of union \n#end // of macro Axis()\n')
        pov_file.write('\n#macro Axisc( AxisLen,Yloc,Zloc, Dark_Texture,Light_Texture)\nunion{\n\tcylinder { <0,0,0>,<0,0,AxisLen>,0.05\n\t\ttexture{checker texture{Dark_Texture }\n\t\t\ttexture{Light_Texture}\n\t\t }\n\t}\n\tcone{<0,0,AxisLen>,0.15,<0,0,AxisLen+0.6>,0\n\t\ttexture{Dark_Texture}\n\t\t}translate<3.8,Yloc,Zloc>\n\t} // end of union \n#end // of macro Axis()\n')
        pov_file.write('\n#macro Axisp( AxisLen,Yloc,Zloc, Dark_Texture,Light_Texture)\nunion{\n\tcylinder { <0,0,0>,<0,0,AxisLen>,0.15\n\t\ttexture{checker texture{Dark_Texture }\n\t\t\ttexture{Light_Texture}\n\t\t }\n\t}\n\tcone{<0,AxisLen,0>,0.2,<0,AxisLen+1.2,0>,0\n\t\ttexture{Dark_Texture}\n\t\t}translate<3.8,Yloc,Zloc>\n\t} // end of union \n#end // of macro Axis()\n')
        pov_file.write('\n#macro AxisXYZ( AxisLenX, AxisLenY, AxisLenZ, Tex_Dark, Tex_Light)\n//--------------------- drawing of 3 Axes --------------------------------\nunion{\n#if (AxisLenX != 0)\nobject { Axis_(AxisLenX, Tex_Dark, Tex_Light)   rotate< 0,0,-90>}// x-Axis\ntext   { ttf "arial.ttf",  "y",  0.15,  0  texture{Tex_Dark}\n\trotate< -90,180,-180> scale 0.5 translate <AxisLenX+0.05,-0.4, 0.40>}\n#end // of #if\n')
        pov_file.write('#if (AxisLenY != 0)\n object { Axis_(AxisLenY, Tex_Dark, Tex_Light)   rotate< 0,0,  0>}// y-Axis\ntext   { ttf "arial.ttf",  "x",  0.15,  0  texture{Tex_Dark}\n\trotate<90, 0,90>  scale 0.5 translate <-0.40,AxisLenY+0.20,0.50>}\n#end // of #if\n')
        pov_file.write('#if (AxisLenZ != 0)\nobject { Axis_(AxisLenZ, Tex_Dark, Tex_Light)   rotate<90,0,  0>}// z-Axis\ntext   { ttf "arial.ttf",  "z",  0.15,  0  texture{Tex_Dark}\n\trotate<-90,0,180>  scale 0.5 translate <-0.4,0.0,AxisLenZ+0.10>}\n#end // of #if\n} // end of union\n#end// of macro AxisXYZ( ... )\n')
        pov_file.write('//------------------------------------------------------------------------\n\n#declare Texture_A_Dark  = texture {\n\t\tpigment{color rgb<1,0.25,0>}\n\t\tfinish {ambient 0.15 diffuse 0.85 phong 1}\n\t\t}\n#declare Texture_A_Light = texture {\n\t\tpigment{color rgb<1,1,1>}\n\t\tfinish {ambient 0.15 diffuse 0.85 phong 1}\n\t\t}\n\n')
        pov_file.write('#macro GenSphere(Pos,Radius,Phi,Theta,Psi,Color)\n\tsphere { \n\t <0,0,0> Radius \n\t texture{ pigment { Color } \n\t\t    } \n\tinterior { ior 1.5 }\n\n\ttranslate Pos\n}\n\n#end\n')
        # pov_file.write('box { <0,0,0>, <100,0.0,100 >    // plane with layered textures\n\ttexture { pigment{color rgbt<0.1,0.1,0.1,0.2>}\n\t\tfinish {ambient 1.0 diffuse 0.6}}\n//\ttexture { Raster(RasterScale,RasterHalfLine ) rotate<0,0,0> }\n//\ttexture { Raster(RasterScale,RasterHalfLineZ) rotate<0,90,0>}\n\trotate<0,0,0>\n\tno_shadow\n\t}\n\n')
        # pov_file.write('box { <0,100,0>, <100,100,100 >    // plane with layered textures\n\ttexture { pigment{color rgbt<0.1,0.1,0.1,0.2>}\n\t\tfinish {ambient 1.0 diffuse 0.6 roughness 0.001 \n\treflection{0.75\n\tmetallic} }}\n//\ttexture { Raster(RasterScale,RasterHalfLine ) rotate<0,0,0> }\n//\ttexture { Raster(RasterScale,RasterHalfLineZ) rotate<0,90,0>}\n\trotate<0,0,0>\n\tno_shadow\n\t}\n\n')
        # pov_file.write('disc { <3.72,5,11.5>,<1,0,0>,0.8,0.65\n\t\ttexture{ \n\tpigment{color rgbt<0.5,0.5,0.5,0>} }\n\t}\n\t')

        print(f"num_objects= {num_objects}")

        for _ in range(num_objects):
            
            x,y,z,r= read_floats(data, vector_num)
            # print(f"x= {x}, y= {y}, z= {z}, r= {r}")
            pov_file.write(f"GenSphere (<{x }, {y }, {z }>, {r }, {0 * 180 / 3.141592653589793}, {0 * 180 / 3.141592653589793}, {0 * 180 / 3.141592653589793}, rgb<0.0, 0.2, 0.9>)\n")
            # elif obj_type == 1.0:
            # pov_file.write(f"GenSphere (<{x * 100}, {y * 100}, {z * 100}>, {r * 100}, {phi * 180 / 3.141592653589793}, {theta * 180 / 3.141592653589793}, {psi * 180 / 3.141592653589793}, rgb<1.0, 0.0, 0.0>)\n")



def main():
    print("File conversion started\n\n")

    if len(sys.argv) < 2:
        print("Usage: python script.py <datafile> [options]")
        return

    data_filename = sys.argv[1]

    try:
        data = open(data_filename, 'rb')
        print("Successfully open file.")
    except FileNotFoundError:
        print("Failed to open file.")
        return

    
    num_frames = 0


    while True:
        try:
            num_particles = struct.unpack('f', data.read(float_size))[0]
            data.seek(int(num_particles) * vector_num * float_size,1)
            num_frames += 1
        except:
            print(f"Number of frames: {num_frames}")
            break



    # Parse command-line arguments
    fin = False
    startf = False
    endf = False
    end_time = 0
    start_time = 0
    skip_frames = 1
    frame = 0

    if len(sys.argv) > 2:
        i = 2
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg == "-final":
                fin = True
                print("Rendering final frame.")
            elif arg == "-start":
                start_time = int(sys.argv[i + 1])
                print(f"Starting frame: {start_time}")
                startf = True
                i += 1
            elif arg == "-end":
                end_time = int(sys.argv[i + 1])
                print(f"Ending frame: {end_time}")
                endf = True
                i += 1
            elif arg == "-skip":
                skip_frames = int(sys.argv[i + 1])
                print(f"Skip every: {skip_frames}")
                endf = True
                i += 1
            i += 1

    

    if not endf:
        end_time = num_frames

    if fin:
        frame = num_frames - 2
    elif startf:
        frame = start_time
    else:
        frame = 0

    print(f"Starting from frame {start_time} and ending at frame {end_time}")

    data.seek(0)
    # Skip to the starting frame
    for start_frame in range(frame):
        num_particles = struct.unpack('f', data.read(float_size))[0]
        data.seek(int(num_particles) * vector_num * float_size, 1)

    # Process frames
    while  frame < end_time:
        if (frame - start_time) % skip_frames == 0:
            num_particles = struct.unpack('f', data.read(float_size))[0]
            print(f"during run numParticles= {num_particles}")
            write_pov_file_and_generate_frame(data, int(num_particles), frame)

            num = str(frame)
            com = f"povray +Itemp_{num}.pov -W3840 -H2160"
            os.system(com)
        else:
            num_particles = struct.unpack('f', data.read(float_size))[0]
            data.seek(int(num_particles) * vector_num * float_size, 1)
        frame += 1

    data.close()
    print("File conversion success!")

if __name__ == "__main__":
    main()
