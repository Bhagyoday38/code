import java.util.*;

public class UniqueElements {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        
        System.out.println("Enter the array elements: ");
        int[] inputArray = new int[scanner.nextInt()];
        for (int i = 0; i < inputArray.length; i++) {
            inputArray[i] = scanner.nextInt();
        }

        
        Set<Integer> uniqueElements = new HashSet<>();
        for (int element : inputArray) {
            uniqueElements.add(element);
        }

       
        int[] uniqueArray = new int[uniqueElements.size()];
        int index = 0;
        for (int element : uniqueElements) {
            uniqueArray[index++] = element;
        }

        
        System.out.println("Unique array: " + Arrays.toString(uniqueArray));
    }
}
