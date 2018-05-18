#include <iostream.h>
#include <iomanip.h>
#include<stdio.h>
#include <stdlib.h>

template<typename T>
class Vec3{
		public:
			Vec3() : x(T(0)),y(T(0)),z(T(0)){}
			Vec3(T xx) : x(xx),y(xx),z(xx){}
			Vec3(T xx,T yy,T zz): x(xx),y(yy),z(zz){}

			Vec3 operator + (const Vec3 &v) const
			{return Vec3(x+v.x,y+v.y,z+v.z);}
			Vec3 operator - (const Vec3 &v) const
			{return Vec3(x-v.x,y-v.y,z-v.z);}
			Vec3 operator - () const
			{return Vec3(-x,-y,-z);}
			Vec3 operator * (const T &r) const
			{return Vec3(x*r,y*r,z*r);}
			Vec3 operator * (const Vec3 &v) const
			{return Vec3(x*v.x,y*v.y,z*v.z);}
			T dotProduct(const Vec3<T> &v){
				return(x*v.x+y*v.y+z*v.z);
			}
			Vec3& operator *= (const T &r){
				x*=r;y*=r;z*=r;return *this;
			}
			Vec3& operator /= (const T &r){
				x/=r;y/=r;z/=r;return *this;
			}
			Vec3 crossProduct(const Vec3 &v) const{
				return Vec3((y*v.z-z*v.y),(x*v.z-z*v.x)*(-1),(x*v.y-y*v.x));
			}
			T norm() const
			{return(x*x+y*y+z*z);}
			T length() const
			{return sqrt(norm());}

}