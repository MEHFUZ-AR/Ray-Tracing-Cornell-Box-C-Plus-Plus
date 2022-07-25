#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "Image.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "Vect.h"
#include "Ray.h"
#include "Camera.h"
#include "Color.h"
#include "Source.h"
#include "Light.h"
#include "Object.h"
#include "Sphere.h"
#include "Plane.h"
#include "Triangle.h"


using namespace std;

typedef float Scalar;
typedef Eigen::Matrix<Scalar, 3, 1> Vec3;
typedef Eigen::Matrix<Scalar, 2, 1> Vec2;

using Colour = cv::Vec3b; // RGB Value
//Colour black() { return Colour(0, 0, 0); }

//bounding the channel wise pixel color between 0 to 255
uchar Clamp(int color)
{
	if (color < 0) return 0;
	if (color >= 255) return 255;
	return color;
}

struct RGBType {
	double r;
	double g;
	double b;
};



int closestIntersectionPoint(vector<double> object_intersections) {
	// return the index of the winning intersection
	int index_of_minimum_value;

	// prevent unnessary calculations
	if (object_intersections.size() == 0) {
		// if there are no intersections
		return -1;
	}
	else if (object_intersections.size() == 1) {
		if (object_intersections.at(0) > 0) {
			// if that intersection is greater than zero then its our index of minimum value
			return 0;
		}
		else {
			// otherwise the only intersection value is negative
			return -1;
		}
	}
	else {
		// otherwise there is more than one intersection
		// first find the maximum value

		double max = 0;
		for (int i = 0; i < object_intersections.size(); i++) {
			if (max < object_intersections.at(i)) {
				max = object_intersections.at(i);
			}
		}

		// then starting from the maximum value find the minimum positive value
		if (max > 0) {
			// we only want positive intersections
			for (int index = 0; index < object_intersections.size(); index++) {
				if (object_intersections.at(index) > 0 && object_intersections.at(index) <= max) {
					max = object_intersections.at(index);
					index_of_minimum_value = index;
				}
			}

			return index_of_minimum_value;
		}
		else {
			// all the intersections were negative
			return -1;
		}
	}
}

Color getColorAt(Vect intersection_position, Vect intersecting_ray_direction, vector<Object*> scene_objects, int closest_intersection_index, vector<Source*> light_sources, double accuracy, double ambientlight) {

	Color winning_object_color = scene_objects.at(closest_intersection_index)->getColor();
	Vect winning_object_normal = scene_objects.at(closest_intersection_index)->getNormalAt(intersection_position);

	if (winning_object_color.getColorSpecial() == 2) {
		// checkered/tile floor pattern

		int square = (int)floor(intersection_position.getVectX()) + (int)floor(intersection_position.getVectZ());

		if ((square % 2) == 0) {
			// purple tiles
			winning_object_color.setColorRed(0.372);
			winning_object_color.setColorGreen(0.149);
			winning_object_color.setColorBlue(0.470);
		}
		else {
			// white tiles
			winning_object_color.setColorRed(1);
			winning_object_color.setColorGreen(1);
			winning_object_color.setColorRed(1);
		}
	}

	Color final_color = winning_object_color.colorScalar(ambientlight);

	if (winning_object_color.getColorSpecial() > 0 && winning_object_color.getColorSpecial() <= 1) {
		// reflection from objects with specular intensity
		double dot1 = winning_object_normal.dotProduct(intersecting_ray_direction.negative());
		Vect scalar1 = winning_object_normal.vectMult(dot1);
		Vect add1 = scalar1.vectAdd(intersecting_ray_direction);
		Vect scalar2 = add1.vectMult(2);
		Vect add2 = intersecting_ray_direction.negative().vectAdd(scalar2);
		Vect reflection_direction = add2.normalize();

		Ray reflection_ray(intersection_position, reflection_direction);

		// determine what the ray intersects with first
		vector<double> reflection_intersections;

		for (int reflection_index = 0; reflection_index < scene_objects.size(); reflection_index++) {
			reflection_intersections.push_back(scene_objects.at(reflection_index)->findIntersection(reflection_ray));
		}

		int closest_intersection_index_with_reflection = closestIntersectionPoint(reflection_intersections);

		if (closest_intersection_index_with_reflection != -1) {
			// reflection ray missed everthing else
			if (reflection_intersections.at(closest_intersection_index_with_reflection) > accuracy) {
				// determine the position and direction at the point of intersection with the reflection ray
				// the ray only affects the color if it reflected off something

				Vect reflection_intersection_position = intersection_position.vectAdd(reflection_direction.vectMult(reflection_intersections.at(closest_intersection_index_with_reflection)));
				Vect reflection_intersection_ray_direction = reflection_direction;

				Color reflection_intersection_color = getColorAt(reflection_intersection_position, reflection_intersection_ray_direction, scene_objects, closest_intersection_index_with_reflection, light_sources, accuracy, ambientlight);

				final_color = final_color.colorAdd(reflection_intersection_color.colorScalar(winning_object_color.getColorSpecial()));
			}
		}
	}

	for (int light_index = 0; light_index < light_sources.size(); light_index++) {
		Vect light_direction = light_sources.at(light_index)->getLightPosition().vectAdd(intersection_position.negative()).normalize();

		float cosine_angle = winning_object_normal.dotProduct(light_direction);

		if (cosine_angle > 0) {
			// test for shadows
			bool shadowed = false;

			Vect distance_to_light = light_sources.at(light_index)->getLightPosition().vectAdd(intersection_position.negative()).normalize();
			float distance_to_light_magnitude = distance_to_light.magnitude();

			Ray shadow_ray(intersection_position, light_sources.at(light_index)->getLightPosition().vectAdd(intersection_position.negative()).normalize());

			vector<double> secondary_intersections;

			for (int object_index = 0; object_index < scene_objects.size() && shadowed == false; object_index++) {
				secondary_intersections.push_back(scene_objects.at(object_index)->findIntersection(shadow_ray));
			}

			for (int c = 0; c < secondary_intersections.size(); c++) {
				if (secondary_intersections.at(c) > accuracy) {
					if (secondary_intersections.at(c) <= distance_to_light_magnitude) {
						shadowed = true;
					}
					break;
				}
				
			}

			if (shadowed == false) {
				final_color = final_color.colorAdd(winning_object_color.colorMultiply(light_sources.at(light_index)->getLightColor()).colorScalar(cosine_angle));

				if (winning_object_color.getColorSpecial() > 0 && winning_object_color.getColorSpecial() <= 1) {
					// special [0-1]
					double dot1 = winning_object_normal.dotProduct(intersecting_ray_direction.negative());
					Vect scalar1 = winning_object_normal.vectMult(dot1);
					Vect add1 = scalar1.vectAdd(intersecting_ray_direction);
					Vect scalar2 = add1.vectMult(2);
					Vect add2 = intersecting_ray_direction.negative().vectAdd(scalar2);
					Vect reflection_direction = add2.normalize();

					double specular = reflection_direction.dotProduct(light_direction);
					if (specular > 0) {
						specular = pow(specular, 10);
						final_color = final_color.colorAdd(light_sources.at(light_index)->getLightColor().colorScalar(specular * winning_object_color.getColorSpecial()));
					}
				}

			}

		}
	}

	return final_color.clip();
}

int thisone;
vector<Object*> scene_objects;
void Box(Vect corner1, Vect corner2, Color color) {
	// corner 1
	double c1x = corner1.getVectX();
	double c1y = corner1.getVectY();
	double c1z = corner1.getVectZ();

	// corner 2
	double c2x = corner2.getVectX();
	double c2y = corner2.getVectY();
	double c2z = corner2.getVectZ();

	Vect A(c2x, c1y, c1z);
	Vect B(c2x, c1y, c2z);
	Vect C(c1x, c1y, c2z);

	Vect D(c2x, c2y, c1z);
	Vect E(c1x, c2y, c1z);
	Vect F(c1x, c2y, c2z);

	
	scene_objects.push_back(new Triangle(D, A, corner1, color));
	scene_objects.push_back(new Triangle(corner1, E, D, color));

	scene_objects.push_back(new Triangle(corner2, B, A, color));
	scene_objects.push_back(new Triangle(A, D, corner2, color));

	scene_objects.push_back(new Triangle(F, C, B, color));
	scene_objects.push_back(new Triangle(B, corner2, F, color));

	scene_objects.push_back(new Triangle(E, corner1, C, color));
	scene_objects.push_back(new Triangle(C, F, E, color));

	scene_objects.push_back(new Triangle(D, E, F, color));
	scene_objects.push_back(new Triangle(F, corner2, D, color));

	scene_objects.push_back(new Triangle(corner1, A, B, color));
	scene_objects.push_back(new Triangle(B, C, corner1, color));
	


}

void cornell(Vect corner1, Vect corner2, Color color1, Color color2, Color color3) {
	// corner 1
	double c1x = corner1.getVectX();
	double c1y = corner1.getVectY();
	double c1z = corner1.getVectZ();

	// corner 2
	double c2x = corner2.getVectX();
	double c2y = corner2.getVectY();
	double c2z = corner2.getVectZ();

	Vect A(c2x, c1y, c1z);
	Vect B(c2x, c1y, c2z);
	Vect C(c1x, c1y, c2z);

	Vect D(c2x, c2y, c1z);
	Vect E(c1x, c2y, c1z);
	Vect F(c1x, c2y, c2z);

	//left wall
	scene_objects.push_back(new Triangle(D, A, corner1, color1));
	scene_objects.push_back(new Triangle(corner1, E, D, color1));
	//back wall
	scene_objects.push_back(new Triangle(corner2, B, A, color2));
	scene_objects.push_back(new Triangle(A, D, corner2, color2));
	//right wall
	scene_objects.push_back(new Triangle(F, C, B, color1));
	scene_objects.push_back(new Triangle(B, corner2, F, color1));
	//top ceiling
	scene_objects.push_back(new Triangle(D, E, F, color3));
	scene_objects.push_back(new Triangle(F, corner2, D, color3));



}


int main(int argc, char* argv[]) {
	Image image = Image(500, 500);
	cout << "Rendering the image....." << endl;

	clock_t t1, t2;
	t1 = clock();

	int dpi = 72;
	int width = 500;
	int height = 500;
	int n = width * height;
	RGBType* pixels = new RGBType[n];

	const int samplesPerPixel = 5;
	double aathreshold = 0.1;
	double aspectratio = (double)width / (double)height;
	double ambientlight = 0.2;
	double accuracy = 0.00000001;

	Vect O(0, -0.3, 0);
	Vect X(1, 0, 0);
	Vect Y(0, 1, 0);
	Vect Z(0, 0, 1);

	Vect new_sphere_location(1.75, -0.45, -0.9);

	Vect campos(7, 1.5, 0);

	Vect look_at(-8,0, 0);
	Vect diff_btw(campos.getVectX() - look_at.getVectX(), campos.getVectY() - look_at.getVectY(), campos.getVectZ() - look_at.getVectZ());

	Vect camdir = diff_btw.negative().normalize();
	Vect camright = Y.crossProduct(camdir).normalize();
	Vect camdown = camright.crossProduct(camdir);
	Camera scene_cam(campos, camdir, camright, camdown);

	//color are normalized to 0-1, also give value in bgr format
	Color white_light(1.0, 1.0, 1.0, 0);
	Color pretty_green(0.5, 1.0, 0.5, 0.4);
	Color royal_blue(0.5, 0.25, 0.25, 0.2);
	Color tile_floor(1, 1, 1, 2);
	Color gray(0.3 , 0, 0.4, 0);
	Color purple(0.41, 0.28, 0.25, 0.2);
	Color black(0.0, 0.0, 0.0, 0);
	Color red(0.0, 0.0, 0.8, 0.1);
	Color neon_orange(0.121, 0.372, 1.0, 0.2);
	Color silver(0.753, 0.753, 0.753, 0.1);
	Color deep_green(0.5, 1.0, 0.5, 0.1);

	Vect light_position(1, 5, -1);
	Light scene_light(light_position, white_light);
	vector<Source*> light_sources;
	light_sources.push_back(dynamic_cast<Source*>(&scene_light));

	// scene objects
	Sphere scene_sphere(O, 0.7, pretty_green);
	Sphere scene_sphere2(new_sphere_location, 0.5, royal_blue);
	Plane scene_plane(Y, -1, tile_floor);
	
	
	scene_objects.push_back(dynamic_cast<Object*>(&scene_sphere));
	scene_objects.push_back(dynamic_cast<Object*>(&scene_sphere2));
	scene_objects.push_back(dynamic_cast<Object*>(&scene_plane));
	Box(Vect(-2,1,3), Vect(-4, -1, 1), neon_orange);

	//the cornell box
	cornell(Vect(1.8, 0, 4), Vect(-5,5,-4), gray, purple, deep_green);
	


	int thisone, aa_index;
	double xamnt, yamnt;
	double tempRed, tempGreen, tempBlue;

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			thisone = y * width + x;

			// start with a blank pixel
			double tempRed[samplesPerPixel * samplesPerPixel];
			double tempGreen[samplesPerPixel * samplesPerPixel];
			double tempBlue[samplesPerPixel * samplesPerPixel];

			for (int aax = 0; aax < samplesPerPixel; aax++) {
				for (int aay = 0; aay < samplesPerPixel; aay++) {

					aa_index = aay * samplesPerPixel + aax;

					srand(time(0));

					// create the ray from the camera to this pixel
					if (samplesPerPixel == 1) {

						// start with no anti-aliasing
						if (width > height) {
							// the image is wider than it is tall
							xamnt = ((x + 0.5) / width) * aspectratio - (((width - height) / (double)height) / 2);
							yamnt = ((height - y) + 0.5) / height;
						}
						else if (height > width) {
							// the imager is taller than it is wide
							xamnt = (x + 0.5) / width;
							yamnt = (((height - y) + 0.5) / height) / aspectratio - (((height - width) / (double)width) / 2);
						}
						else {
							// the image is square
							xamnt = (x + 0.5) / width;
							yamnt = ((height - y) + 0.5) / height;
						}
					}
					else {
						// anti-aliasing
						if (width > height) {
							// the image is wider than it is tall
							xamnt = ((x + (double)aax / ((double)samplesPerPixel - 1)) / width) * aspectratio - (((width - height) / (double)height) / 2);
							yamnt = ((height - y) + (double)aax / ((double)samplesPerPixel - 1)) / height;
						}
						else if (height > width) {
							// the imager is taller than it is wide
							xamnt = (x + (double)aax / ((double)samplesPerPixel - 1)) / width;
							yamnt = (((height - y) + (double)aax / ((double)samplesPerPixel - 1)) / height) / aspectratio - (((height - width) / (double)width) / 2);
						}
						else {
							// the image is square
							xamnt = (x + (double)aax / ((double)samplesPerPixel - 1)) / width;
							yamnt = ((height - y) + (double)aax / ((double)samplesPerPixel - 1)) / height;
						}
					}

					Vect cam_ray_origin = scene_cam.getCameraPosition();
					Vect cam_ray_direction = camdir.vectAdd(camright.vectMult(xamnt - 0.5).vectAdd(camdown.vectMult(yamnt - 0.5))).normalize();

					Ray cam_ray(cam_ray_origin, cam_ray_direction);

					vector<double> intersections;

					for (int index = 0; index < scene_objects.size(); index++) {
						intersections.push_back(scene_objects.at(index)->findIntersection(cam_ray));
					}

					int closest_intersection_index = closestIntersectionPoint(intersections);

					if (closest_intersection_index == -1) {
						// set the backgroung black
						tempRed[aa_index] = 1;
						tempGreen[aa_index] = 1;
						tempBlue[aa_index] = 1;
					}
					else {
						// index coresponds to an object in our scene
						if (intersections.at(closest_intersection_index) > accuracy) {
							// determine the position and direction vectors at the point of intersection

							Vect intersection_position = cam_ray_origin.vectAdd(cam_ray_direction.vectMult(intersections.at(closest_intersection_index)));
							Vect intersecting_ray_direction = cam_ray_direction;

							Color intersection_color = getColorAt(intersection_position, intersecting_ray_direction, scene_objects, closest_intersection_index, light_sources, accuracy, ambientlight);

							tempRed[aa_index] = intersection_color.getColorRed();
							tempGreen[aa_index] = intersection_color.getColorGreen();
							tempBlue[aa_index] = intersection_color.getColorBlue();
						}
					}
				}
			}

			// average the pixel color
			double totalRed = 0;
			double totalGreen = 0;
			double totalBlue = 0;

			for (int iRed = 0; iRed < samplesPerPixel * samplesPerPixel; iRed++) {
				totalRed = totalRed + tempRed[iRed];
			}
			for (int iGreen = 0; iGreen < samplesPerPixel * samplesPerPixel; iGreen++) {
				totalGreen = totalGreen + tempGreen[iGreen];
			}
			for (int iBlue = 0; iBlue < samplesPerPixel * samplesPerPixel; iBlue++) {
				totalBlue = totalBlue + tempBlue[iBlue];
			}

			double avgRed = totalRed / (samplesPerPixel * samplesPerPixel);
			double avgGreen = totalGreen / (samplesPerPixel * samplesPerPixel);
			double avgBlue = totalBlue / (samplesPerPixel * samplesPerPixel);

			pixels[thisone].r = float(avgRed * 255);
			pixels[thisone].g = float(avgGreen * 255);
			pixels[thisone].b = float(avgBlue * 255);

			Colour colur(0.0, 0.0, 0.0);
			colur[0] = pixels[thisone].r;
			colur[1] = pixels[thisone].g;
			colur[2] = pixels[thisone].b;
			image(499 - y, 499 - x) = colur;

		}
	}

	image.save("./result.png");
	image.display();

	delete pixels, tempRed, tempGreen, tempBlue;

	t2 = clock();
	float diff = ((float)t2 - (float)t1) / 1000;

	cout << diff << " seconds" << endl;

	return EXIT_SUCCESS;
}
