import math

def slots_data_finder(prediction):
    """
    Generates manuever points for the parking according to the parking type (parallel or perpendicular)

    Parameters:
    ------------
    prediction: list
        list of prediction as example
        [{'boxes': tensor([[789.0319, 591.6688, 898.3337, 768.5793]]), 'labels': tensor([1]), 'scores': tensor([0.9998])}]


    Returns:
    ------------
    list
        A list of poses including the start and end pose.
    """
    
    #adjust the pixels to meter according to lidar scale
    lidar_scale = 0.05

    # Car constants in meters, change according to your car, but also divided with lidar_scale
    Lro = 0.05 / lidar_scale # rear overhang
    Lfo = 0.05 / lidar_scale  # front overhang
    Rmin = 0.5  / lidar_scale # minimum turn radius
    Lwb = 0.5  / lidar_scale # wheelbase length
    Ls = 0.1 / lidar_scale  # Safe distance between vehicle and the boundary of parking spot
    W = 0.15 / lidar_scale  #car width
    L = 0.4 / lidar_scale #car length


    img_shape = prediction[1]




    slots = []

    boxes = prediction[0]['boxes']
    for idx,box in enumerate(boxes):
        
        if prediction[0]['scores'][idx] < 0.5:
            continue
        


        xmin, ymin, xmax, ymax = box
        box_width = xmax - xmin
        box_height = ymax - ymin 


        if box_height > box_width:
            #this is parallel
            parking_type = 0
        else:
            #this is perpendicular
            parking_type = 1


        #this is parallel
        if parking_type == 0:
            
            # Calculate minimum parallel slot size 
            # Equation (1)
            Wp_parallel = W + 2 * Ls
            # yB = Lp

            # Equation (3)
            xB1 = Lro + Ls + math.sqrt((2 * Rmin - Ls) * (W + Ls) + (L - Lro) ** 2)

            # Equation (4)
            xB2 = Lro + Ls + math.sqrt(2) * Rmin - math.sqrt(((2 - math.sqrt(2)) * Rmin - W - Ls) * (math.sqrt(2) * Rmin + Ls))

            # Equation (2)
            Lp_parallel = min(xB1, xB2)
            # xB = Wp

            # Check if the box found can be used
            if box_height >= Lp_parallel and box_width >= Wp_parallel:
                pass
            else:
                raise ValueError("box is not large enough: ", box_height*lidar_scale,",",box_width*lidar_scale
                                ,"\n but required at least: ", Lp_parallel*lidar_scale,Wp_parallel*lidar_scale
                                 )
            
            poses = {}

            ye = ymax - Ls - Lro
            xe = xmax - box_width/2


            ys = ye - math.sqrt(2) * Rmin
            xs = xe - box_width
            
            poses['start'] = [xs,ys,math.pi/2]
            poses['mid']= [(xe+xs)/2, (ye+ys)/2, math.pi/4]
            poses['end'] = [xe, ye,math.pi/2]
            

        #this is perpendicular            
        elif parking_type == 1:

            # Calculate minimum size for perpendicular parking slot
            Lp_perpendicular = L + Ls
            Wp_perpendicular = math.sqrt(((Rmin + W / 2) ** 2) + Lro ** 2) - (2 * math.sqrt(2) * Rmin - W) / 4
            
            if box_height >= Wp_perpendicular and box_width >= Lp_perpendicular:
                pass
            else:
                raise ValueError("box is not large enough: ", box_height*lidar_scale,",",box_width*lidar_scale
                                ,"\n but required at least: ", Lp_parallel*lidar_scale,Wp_parallel*lidar_scale
                                 )
            poses = {}
            xe = xmax - Lro - Ls
            ye = ymax - box_width / 2

            Lz = box_height - Ls - Lro
            xs = xe - Rmin - Lz
            ys = ye - Rmin

            poses['start'] = [xs,ys,math.pi/2]
            poses['mid']= [(xe+xs)/2, (ye+ys)/2, 3 * math.pi/4]
            poses['end'] = [xe, ye,math.pi]


        slots.append(poses) 

    return slots

