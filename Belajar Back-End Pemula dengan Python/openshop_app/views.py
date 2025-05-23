from django.http import Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import Product
from .serializers import ProductSerializer

class ProductList(APIView):
    def post(self, request):
        serializer = ProductSerializer(data=request.data, context={'request': request})
        if serializer.is_valid(raise_exception=True):
            product = serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request):
        name = request.query_params.get('name', None)
        location = request.query_params.get('location', None)

        products = Product.objects.all()

        if name:
            products = products.filter(name__icontains=name)
        if location:
            products = products.filter(location__icontains=location)

        serializer = ProductSerializer(products, many=True, context={'request': request})
        return Response({
            "products": serializer.data,
        }, status=status.HTTP_200_OK)

class ProductDetail(APIView):
    def get_object(self, pk):
        try:
            return Product.objects.get(pk=pk)
        except Product.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        product = self.get_object(pk)
        serializer = ProductSerializer(product)
        return Response(serializer.data)

    def put(self, request, pk):
        product = self.get_object(pk)
        serializer = ProductSerializer(product, data=request.data, context={'request': request})
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
            product = self.get_object(pk)
            product.is_delete = True
            product.save()
            return Response(status=status.HTTP_204_NO_CONTENT)
