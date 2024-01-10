
    
    # def _get_largest_inscribed_rectangle(
    #     self, 
    #     canvas_size: np.ndarray, 
    #     coordinates: np.ndarray, 
    #     lir_rotation_degree_interval: float
    # ) -> np.ndarray:
        
    #     polygon = Polygon(coordinates)
    #     rotation_anchor = polygon.centroid
        
    #     lir = Polygon()

    #     rotation_degree = 0
    #     while rotation_degree < 360:
            
    #         rotated_random_polygon = affinity.rotate(
    #             geom=polygon, angle=rotation_degree, origin=rotation_anchor
    #         )

    #         binary_grid_shaped_polygon = self._get_binary_grid_shaped_polygon(
    #             coordinates=np.array(rotated_random_polygon.boundary.coords).astype(np.int32),
    #             canvas_size=canvas_size,
    #         )
            
    #         each_lir = self._get_each_lir(binary_grid_shaped_polygon)
            
    #         each_lir_polygon = Polygon(utils.vectorize_polygon_from_array(each_lir))
            
    #         inverted_each_lir_polygon = affinity.rotate(
    #             geom=each_lir_polygon, angle=-rotation_degree, origin=rotation_anchor
    #         )
            
    #         if lir.area < inverted_each_lir_polygon.area:
    #             lir = inverted_each_lir_polygon
            
            
    #         # rotated_coordinates = random_coordinates @ self._get_rotation_matrix(rotation_degree)
    #         # fitted_coordinates = self._get_fitted_coordinates(rotated_coordinates)
    #         # print(Polygon(fitted_coordinates).area)

            
    #         # lir_args = [binary_grid_shaped_polygon, rotation_degree]
    #         # lir = self._get_each_lir(lir_args)
            
    #         # diff = lir - binary_grid_shaped_polygon
            
    #         # vectorized_polygon = utils.vectorize_polygon_from_array(binary_grid_shaped_polygon)
    #         # vectorized_lir = utils.vectorize_polygon_from_array(lir)
            
    #         # rotated_each_polygon = vectorized_polygon @ self._get_rotation_matrix(-rotation_degree)
    #         # rotated_each_lir = vectorized_lir @ self._get_rotation_matrix(-rotation_degree)

    #         # rotated_each_lir = rotated_each_lir - np.array([self.canvas_w_h, self.canvas_w_h])
    #         # rotated_each_lir = rotated_each_lir + np.array([self.canvas_w_h, self.canvas_w_h])

    #         # fitted_polygon = self._get_fitted_coordinates(rotated_each_polygon)
    #         # print(Polygon(fitted_polygon).area)
            
    #         # canvas = np.zeros((*self.canvas_size, 3), dtype=np.uint8)
    #         # cv2.polylines(canvas, [fitted_polygon], isClosed=True, color=(255, 0, 0), thickness=1)
    #         # cv2.polylines(canvas, [rotated_each_lir.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)

            
    #         # color_grid = cv2.cvtColor(diff.astype(np.float32), cv2.COLOR_GRAY2BGR)

    #         # # Assign colors (here, white for 1, black for 0)
    #         # color_grid[diff == 1] = [255, 255, 255]  # White
    #         # color_grid[diff == 0] = [0, 0, 0]        # Black

    #         # Display the image
    #         # cv2.imshow('Binary Grid', canvas)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()

    #         rotation_degree += lir_rotation_degree_interval
        
    #     # with Pool(processes=cpu_count()) as pool:
    #     #     x_squared = pool.starmap(self._get_each_lir, lir_args)
            
    #     #     self._get_each_lir()
        
    #     return