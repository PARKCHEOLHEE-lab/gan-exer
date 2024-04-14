# import commonutils
# import shapely.ops
# import numpy as np
# import shapely.affinity

# from typing import List, Tuple
# from polygonSplitGCN.config import Configuration
# from shapely.geometry import Point, MultiPoint, Polygon, LineString, LinearRing, CAP_STYLE


# class DataCreatorHelper:
#     @staticmethod
#     def divide_linestring(linestring: LineString, count_to_divide: int) -> List[Point]:
#         """_summary_

#         Args:
#             linestring (LineString): _description_
#             count_to_divide (int): _description_

#         Returns:
#             List[Point]: _description_
#         """

#         linestring_coordinates = np.array(linestring.coords)

#         assert len(linestring_coordinates) == 2, "Only can straight linestring be divided."

#         return

#     @staticmethod
#     def extend_linestring(linestring: LineString, start: float, end: float) -> LineString:
#         """Extend a given linestring by the given `start` and `end` values

#         Args:
#             linestring (LineString): linestring
#             start (float): start value
#             end (float): end value

#         Returns:
#             LineString: extended linestring
#         """

#         linestring_coordinates = np.array(linestring.coords)

#         assert len(linestring_coordinates) == 2, "Only can straight linestring be extended."

#         a, b = linestring_coordinates

#         ab = b - a
#         ba = a - b

#         ab_normalized = ab / np.linalg.norm(ab)
#         ba_normalized = ba / np.linalg.norm(ba)

#         a_extended = a + ba_normalized * start
#         b_extended = b + ab_normalized * end

#         extended_linestring = LineString([a_extended, b_extended])

#         assert np.isclose(extended_linestring.length, linestring.length + start + end), "Extension failed."

#         return extended_linestring

#     @staticmethod
#     def compute_polyon_degrees(polygon: Polygon) -> List[float]:
#         """Compute polygon degrees

#         Args:
#             polygon (Polygon): polygon

#         Returns:
#             List[float]: polygon degrees
#         """

#         exterior_coordinates = polygon.exterior.coords[:-1]

#         polygon_degrees = []

#         for ci in range(len(exterior_coordinates)):
#             a = np.array(exterior_coordinates[ci - 1])
#             b = np.array(exterior_coordinates[ci])
#             c = np.array(exterior_coordinates[(ci + 1) % len(exterior_coordinates)])

#             ab = b - a
#             bc = c - b

#             cos_theta = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
#             angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
#             angle_degrees = np.degrees(angle_radians)

#             polygon_degrees.append(angle_degrees)

#         return polygon_degrees

#     @staticmethod
#     def simplify_polygon(polygon: Polygon, tolerance_degree: float = Configuration.TOLEARNCE_DEGREE) -> Polygon:
#         """Simplify a given polygon by removing vertices

#         Args:
#             polygon (Polygon): polygon
#             tolerance_degree (float, optional): threshold to remove. Defaults to 1.0.

#         Returns:
#             Polygon: _description_
#         """

#         exterior_coordinates = polygon.exterior.coords
#         if np.allclose(exterior_coordinates[0], exterior_coordinates[-1]):
#             exterior_coordinates = exterior_coordinates[:-1]

#         polygon_degrees = DataCreatorHelper.compute_polyon_degrees(polygon)

#         assert len(exterior_coordinates) == len(polygon_degrees), "Lengths condition is not satisfied."

#         simplified_coordinates = []
#         for degree, coord in zip(polygon_degrees, exterior_coordinates):
#             if degree > tolerance_degree:
#                 simplified_coordinates.append(coord)

#         return Polygon(simplified_coordinates)

#     @staticmethod
#     def explode_polygon(polygon: Polygon) -> List[LineString]:
#         """Explode a given polygon into a list of LineString objects.

#         Args:
#             polygon (Polygon): polygon

#         Returns:
#             List[LineString]: polygon segments
#         """

#         return [
#             LineString([polygon.exterior.coords[ci], polygon.exterior.coords[ci + 1]])
#             for ci in range(len(polygon.exterior.coords) - 1)
#         ]

#     @staticmethod
#     def triangulate(polygon: Polygon):
#         return


# class DataCreator(DataCreatorHelper):
#     def __init__(self, is_debug_mode: bool = False):
#         self.is_debug_mode = is_debug_mode

#         if self.is_debug_mode:
#             commonutils.add_debugvisualizer(globals())

# def get_triangulations(
#     archi_line: Polygon,
#     simplify: bool,
#     thresh: int = None,
#     includes_intersects_pts: bool = True
# ) -> Tuple[List[Polygon], List[LineString]]:
#     """Get triangulations related geometries

#     Args:
#         archi_line (Polygon): site polygon
#         simplify (bool): switch to simplify
#         thresh (int, optional): baseline length to devide. Defaults to None.
#         includes_intersects_pts (bool, optional): switch to add intersected points. Defaults to True.

#     Returns:
#         Tuple[List[Polygon], List[LineString]]: results related to triangulations
#     """

#     simplified_archiline = DataCreatorHelper.simplify_polygon(archi_line, 0.3) if simplify else archi_line

#     if thresh is not None:
#         simplified_archiline_segs = DataCreatorHelper.explode_polygon(simplified_archiline)

#         all_vertices = []
#         for seg in simplified_archiline_segs:
#             if seg.length > thresh:

# divided_pts = list(
#     divide_segment_arr_to_points_arr(np.array(seg.coords), np.ceil(seg.length / thresh))
# )
#                 all_vertices.extend(divided_pts)

#             else:
#                 all_vertices.extend(np.array(seg.coords))

#         simplified_archiline = Polygon(all_vertices)

#     if includes_intersects_pts:
#         intersects_pts_to_include = []
#         simplified_archiline_segs = DataCreatorHelper.explode_polygon(simplified_archiline)
#         simplified_archiline_vertices = MultiPoint(simplified_archiline.boundary.coords)

#         for seg in simplified_archiline_segs:
#             extended_seg = shapely.affinity.scale(seg, 100, 100)

#             ipts = extended_seg.intersection(simplified_archiline.boundary)
#             if ipts.is_empty:
#                 continue

#             if isinstance(ipts, MultiPoint):
#                 ipts = list(ipts.geoms)
#             else:
#                 ipts = [ipts]

#             for ipt in ipts:
#                 if not simplified_archiline_vertices.buffer(consts.TOLERANCE_LARGE).contains(ipt):
#                     if isinstance(ipt, Point):
#                         intersects_pts_to_include.append(ipt)

#         included_intersects_pts_archiline = list(simplified_archiline_vertices.geoms)

#         curr_vi = 0
#         while curr_vi < len(included_intersects_pts_archiline):

#             next_vi = (curr_vi + 1) % len(included_intersects_pts_archiline)

#             curr_vertex = included_intersects_pts_archiline[curr_vi]
#             next_vertex = included_intersects_pts_archiline[next_vi]

#             curr_segment = LineString([curr_vertex, next_vertex])
#             curr_segment_buffered = curr_segment.buffer(consts.TOLERANCE, cap_style=CAP_STYLE.square)

#             is_inserted = False
#             if curr_segment_buffered.intersects(MultiPoint(intersects_pts_to_include)):
#                 for i, ipt in enumerate(intersects_pts_to_include):
#                     if curr_segment_buffered.contains(ipt):
#                         included_intersects_pts_archiline.insert(next_vi, intersects_pts_to_include.pop(i))
#                         is_inserted = True
#                         break

#             if is_inserted:
#                 continue

#             curr_vi += 1

#         simplified_archiline = Polygon(included_intersects_pts_archiline)

#     triangulations = shapely.ops.triangulate(simplified_archiline)
#     triangulations_filtered_by_area = [tri for tri in triangulations if tri.area >= simplified_archiline.area * 0.01]
#     triangulations_edges = []

#     for tri in triangulations_filtered_by_area:
#         for e in DataCreatorHelper.explode_polygon(tri):
#             if DataCreatorHelper.extend_linestring(
#                 e, -consts.TOLERANCE_MARGIN, -consts.TOLERANCE_MARGIN
#             ).within(simplified_archiline):

#                 is_already_existing = False
#                 for other_e in triangulations_edges:
#                     if e.equals(other_e):
#                         is_already_existing = True
#                         break

#                 if not is_already_existing:
#                     triangulations_edges.append(e)

#     return triangulations_filtered_by_area, triangulations_edges


# def get_site_splits(
#     archi_line: Polygon,
#     thresh: float = 5.0,
#     number_to_split: int = 2,
#     simplify: bool = False,
#     includes_intersects_pts: bool = True,
#     even_area_weight: float = 0.34,
#     ombr_ratio_weight: float = 0.67,
#     slope_similarity_weight: float = 0.045,
# ) -> Tuple[List[Polygon], List[LineString], List[LineString]]:
#     """Get splitted polygons by the given `number_to_split` using triangulation

#     Args:
#         archi_line (Polygon): site polygon
#         thresh (float, optional): baseline length to devide. Defaults to 5.0.
#         number_to_split (int, optional): number to split. Defaults to 2.
#         simplify (bool, optional): swtich to simplify. Defaults to False.
#         includes_intersects_pts (bool, optional): switch to add intersected points. Defaults to True.
#         even_area_weight (float, optional): even area weight for selecting splits candidate. Defaults to 0.34.
#         ombr_ratio_weight (float, optional): ombr ratio weight for selecting splits candidate. Defaults to 0.67.
#         slope_similarity_weight (float, optional): slope weight for selecting splits candidate. Defaults to 0.045.

#     Returns:
#         Tuple[List[Polygon], List[LineString], List[LineString]]: results related to splits
#     """

#     _, triangulations_edges = get_triangulations(
#         archi_line, simplify=simplify, thresh=thresh, includes_intersects_pts=includes_intersects_pts
#     )

#     splitters_selceted = None
#     splits_selected = None
#     splits_score = None

#     for splitters in list(itertools.combinations(triangulations_edges, number_to_split - 1)):

#         exterior_with_splitters = shapely.ops.unary_union(list(splitters) + explode_to_segments(archi_line.boundary))
#         exterior_with_splitters = set_precision(exterior_with_splitters)
#         exterior_with_splitters = shapely.ops.unary_union(exterior_with_splitters)

#         splits = list(shapely.ops.polygonize(exterior_with_splitters))

#         if len(splits) != number_to_split:
#             continue

#         if any(split.area < archi_line.area * 0.25 for split in splits):
#             continue

#         is_acute_angle_in = False
#         is_triangle_shape_in = False
#         for split in splits:
#             split_segments = explode_to_segments(split.boundary)
#             splitter_indices = []

#             for ssi, split_segment in enumerate(split_segments):
#                 reduced_split_segment = DataCreatorHelper.extend_linestring(
#                     split_segment, -consts.TOLERANCE, -consts.TOLERANCE
#                 )
#                 buffered_split_segment = reduced_split_segment.buffer(consts.TOLERANCE, cap_style=CAP_STYLE.flat)

#                 if buffered_split_segment.intersects(MultiLineString(splitters)):
#                     splitter_indices.append(ssi)
#                     splitter_indices.append(ssi + 1)

#             if (np.array([np.degrees(a) for a in angle_of_polygon_vertices(split)])[splitter_indices] < 20).sum():
#                 is_acute_angle_in = True
#                 break

#             if len(explode_to_segments(simplify_polygon(split).boundary)) == 3:
#                 is_triangle_shape_in = True
#                 break

#         if is_acute_angle_in or is_triangle_shape_in:
#             continue

#         sorted_splits_area = sorted([split.area for split in splits], reverse=True)
#         even_area_score = (sorted_splits_area[0] - sum(sorted_splits_area[1:])) / archi_line.area * even_area_weight

#         ombr_ratio_scores = []
#         slope_similarity_scores = []

#         for split in splits:
#             ombr = split.minimum_rotated_rectangle
#             each_ombr_ratio = split.area / ombr.area
#             inverted_ombr_score = 1 - each_ombr_ratio
#             ombr_ratio_scores.append(inverted_ombr_score)

#             slopes = []
#             for splitter in splitters:
#                 if split.buffer(consts.TOLERANCE).intersects(splitter):
#                     slopes.append(compute_slope(Point(splitter.coords[0]), Point(splitter.coords[1])))

#             splitter_main_slope = max(slopes, key=abs)

#             split_slopes_similarity = []
#             split_segments = explode_to_segments(split.boundary)
#             for split_seg in split_segments:
#                 split_seg_slope = compute_slope(Point(split_seg.coords[0]), Point(split_seg.coords[1]))
#                 split_slopes_similarity.append(abs(splitter_main_slope - split_seg_slope))

#             avg_slope_similarity = sum(split_slopes_similarity) / len(split_slopes_similarity)
#             slope_similarity_scores.append(avg_slope_similarity)

#         ombr_ratio_score = abs(ombr_ratio_scores[0] - sum(ombr_ratio_scores[1:])) * ombr_ratio_weight
#         slope_similarity_score = sum(slope_similarity_scores) / len(splits) * slope_similarity_weight

#         score_sum = even_area_score + ombr_ratio_score + slope_similarity_score

#         if splits_score is None or splits_score > score_sum:
#             splits_score = score_sum
#             splits_selected = splits
#             splitters_selceted = splitters

#     return splits_selected, triangulations_edges, splitters_selceted


# if __name__ == "__main__":

#     is_debug_mode = True
#     if is_debug_mode:
#         commonutils.add_debugvisualizer(globals())

#     import numpy as np
#     np.random.seed(0)

#     def _get_random_coordinates(
#         vertices_count_min: int, vertices_count_max: int, scale_factor: float = 1.0
#     ) -> np.ndarray:
#         """Generate non-intersected polygon randomly

#         Args:
#             vertices_count_min (int): random vertices count minimum value
#             vertices_count_max (int): random vertices count maximum value
#             scale_factor (float, optional): constant to scale. Defaults to 1.0.

#         Returns:
#             np.ndarray: random coordinates
#         """

#         vertices_count = np.random.randint(vertices_count_min, vertices_count_max)
#         vertices = np.random.rand(vertices_count, 2)
#         vertices_centroid = np.mean(vertices, axis=0)

#         coordinates = sorted(vertices, key=lambda p, c=vertices_centroid: np.arctan2(p[1] - c[1], p[0] - c[0]))

#         coordinates = np.array(coordinates)
#         coordinates[:, 0] *= scale_factor
#         coordinates[:, 1] *= scale_factor

#         return coordinates

#     p_ = Polygon(_get_random_coordinates(100, 200)).convex_hull
#     p = DataCreatorHelper.simplify_polygon(
#         polygon=p_,
#         tolerance_degree=1
#     )

#     p_ = Polygon(_get_random_coordinates(100, 200))
#     p = DataCreatorHelper.simplify_polygon(
#         polygon=p_,
#         tolerance_degree=1
#     )

#     p_ = Polygon(_get_random_coordinates(100, 200)).convex_hull
#     p = DataCreatorHelper.simplify_polygon(
#         polygon=p_,
#         tolerance_degree=1
#     )

#     p_ = Polygon(_get_random_coordinates(100, 200))
#     p = DataCreatorHelper.simplify_polygon(
#         polygon=p_,
#         tolerance_degree=1
#     )

#     l_ = LineString([[0,0], [15, 3]])
#     l = DataCreatorHelper.extend_linestring(l_, start=0.2, end=0.8)

#     l_ = LineString([[-5,3], [15, 3]])
#     l = DataCreatorHelper.extend_linestring(l_, start=-20, end=0)

#     print()
