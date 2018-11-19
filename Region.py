import math

class Region:

    def __init__(self, x_max:int, x_min:int, y_max:int, y_min:int) -> None:
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min

    def get_radius(self) -> int:
        return int(math.sqrt(math.pow(self.x_max - self.x_min, 2) + math.pow(self.y_max - self.y_min, 2)) / 2)

    def get_center_point(self) -> [int,int]:
        return [int(self.x_min + (self.x_max - self.x_min) / 2), int(self.y_min + (self.y_max - self.y_min) / 2)]

    def try_combine(self, potential_region) -> bool:
        combine_region = False

        x_center, y_center = self.get_center_point()
        x_center_potential, y_center_potential = potential_region.get_center_point()
        
        distance_between_centers = math.sqrt(math.pow(abs(x_center - x_center_potential), 2) + math.pow(abs(y_center - y_center_potential), 2))

        # If potential region has common area
        if(distance_between_centers <= self.get_radius() + potential_region.get_radius()):
            combine_region = True
        # If potential region is close enough
        elif(distance_between_centers - self.get_radius() <= 10):
            combine_region = True
        
        if(combine_region):
            self.x_max = max(self.x_max, potential_region.x_max)
            self.x_min = min(self.x_min, potential_region.x_min)
            self.y_max = max(self.y_max, potential_region.y_max)
            self.y_min = min(self.y_min, potential_region.y_min)

        return combine_region