import os


class Rack:
    def __init__(self, name='rack', storey_num=3, line_num=5, size_h=1600, size_w=1080, start_point_y=0, start_point_x=0,
                 start_slot_y=0, start_slot_x=14.57, shift_line=200, shift_storey=70, slot_h=120, slot_w=110.85):
        self.name = name
        self.storey_num = storey_num
        self.line_num = line_num
        self.size = (size_h, size_w)
        self.start_point = (start_point_y, start_point_x)
        self.start_slot = (start_slot_y, start_slot_x)
        self.shift_line = shift_line
        self.shift_storey = shift_storey
        self.slot_size = (slot_h, slot_w)

    def change_storey_num(self, storey_num):
        self.storey_num = storey_num

    def get_storey_num(self):
        return self.storey_num

    def change_line_num(self, line_num):
        self.line_num = line_num

    def get_line_num(self):
        return self.line_num

    def change_size(self, size_h, size_w):
        self.size = (size_h, size_w)

    def get_size(self):
        return self.size

    def change_start_point(self, start_point_y, start_point_x):
        self.start_point = (start_point_y, start_point_x)

    def get_start_point(self):
        return self.start_point

    def change_start_slot(self, start_slot_y, start_slot_x):
        self.start_slot = (start_slot_y, start_slot_x)

    def get_start_slot(self):
        return self.start_slot

    def change_shift_line(self, shift_line):
        self.shift_line = shift_line

    def get_shift_line(self):
        return self.shift_line

    def change_shift_storey(self, shift_storey):
        self.shift_storey = shift_storey

    def get_shift_storey(self):
        return self.shift_storey

    def change_slot_size(self, slot_h, slot_w):
        self.slot_size = (slot_h, slot_w)

    def get_slot_size(self):
        return self.slot_size


class Sampler:
    def __init__(self, name='sampler', inner_diameter=29, outer_diameter=42, outer_ring_d=61, back_ring_d=16,
                 front_part_l=1070, mid_part_l=312, back_part_l=18):
        self.name = name
        self.inner_diameter = inner_diameter
        self.outer_diameter = outer_diameter
        self.outer_ring_d = outer_ring_d
        self.back_ring_d = back_ring_d
        self.front_part_l = front_part_l
        self.mid_part_l = mid_part_l
        self.back_part_l = back_part_l

    def change_inner_diameter(self, inner_diameter):
        self.inner_diameter = inner_diameter

    def get_inner_diameter(self):
        return self.inner_diameter

    def change_outer_diameter(self, outer_diameter):
        self.outer_diameter = outer_diameter

    def get_outer_diameter(self):
        return self.outer_diameter

    def change_outer_ring_d(self, outer_ring_d):
        self.outer_ring_d = outer_ring_d

    def get_outer_ring_d(self):
        return self.outer_ring_d

    def change_back_ring_d(self, back_ring_d):
        self.back_ring_d = back_ring_d

    def get_back_ring_d(self):
        return self.back_ring_d

    def change_front_part_l(self, front_part_l):
        self.front_part_l = front_part_l

    def get_front_part_l(self):
        return self.front_part_l

    def change_mid_part_l(self, mid_part_l):
        self.mid_part_l = mid_part_l

    def get_mid_part_l(self):
        return self.mid_part_l

    def change_back_part_l(self, back_part_l):
        self.back_part_l = back_part_l

    def get_back_part_l(self):
        return self.back_part_l

    def get_length(self):
        return self.front_part_l+self.mid_part_l+self.back_part_l


class Probe:
    def __init__(self):
        pass


class End:
    def __init__(self):
        pass
