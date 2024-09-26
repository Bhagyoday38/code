import java.util.Scanner;

public class MatrixMultiplication {

    public static void main(String[] args) {
        
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter the number of rows in the first matrix:");
        int numRows1 = scanner.nextInt();
        System.out.println("Enter the number of columns in the first matrix:");
        int numCols1 = scanner.nextInt();
        System.out.println("Enter the elements of the first matrix:");
        int[][] matrix1 = new int[numRows1][numCols1];
        for (int i = 0; i < numRows1; i++) {
            for (int j = 0; j < numCols1; j++) {
                matrix1[i][j] = scanner.nextInt();
            }
        }
        System.out.println("Enter the number of rows in the second matrix:");
        int numRows2 = scanner.nextInt();
        System.out.println("Enter the number of columns in the second matrix:");
        int numCols2 = scanner.nextInt();
        System.out.println("Enter the elements of the second matrix:");
        int[][] matrix2 = new int[numRows2][numCols2];
        for (int i = 0; i < numRows2; i++) {
            for (int j = 0; j < numCols2; j++) {
                matrix2[i][j] = scanner.nextInt();
            }
        }

        
        if (numCols1 != numRows2) {
            System.out.println("The matrices cannot be multiplied.");
            return;
        }

        
        int[][] productMatrix = new int[numRows1][numCols2];
        for (int i = 0; i < numRows1; i++) {
            for (int j = 0; j < numCols2; j++) {
                for (int k = 0; k < numCols1; k++) {
                    productMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
